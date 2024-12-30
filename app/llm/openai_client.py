import aiohttp
from typing import Any, Dict, List, Optional
import logging
import json
from datetime import datetime
import os
from dotenv import load_dotenv
import asyncio
import time
from http import HTTPStatus
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class OpenAIError(Exception):
    """Custom exception class for OpenAI-related errors."""

    def __init__(
        self,
        message: str,
        error_type: str = "api_error",
        details: Optional[Dict] = None,
    ):
        self.message = message
        self.error_type = error_type
        self.details = details or {}
        super().__init__(self.message)


class OpenAIResumeEnhancer:
    def __init__(self):
        """
        Initialize the OpenAIResumeEnhancer with environment variables and API configuration.
        """
        # Load environment variables
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("OPENAI_API_KEY not found in environment variables")
            raise OpenAIError(
                message="OpenAI API key not found in environment variables",
                error_type="configuration_error",
            )

        # Initialize OpenAI client
        try:
            self.client = AsyncOpenAI(api_key=self.api_key)
            logger.info("Successfully initialized OpenAI client")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise OpenAIError(
                message=f"Failed to initialize OpenAI client: {str(e)}",
                error_type="initialization_error",
            )

        # Configuration
        self.model = "gpt-4o-2024-11-20"  # Latest GPT-4 model
        self.max_tokens = 4096
        self.temperature = 0.7

        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 0.1  # 100ms between requests

    async def test_connection(self) -> Dict[str, Any]:
        """Test the connection to OpenAI API."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": "Hello, this is a test message. Please respond with 'Connection successful.'",
                    },
                ],
                max_tokens=100,
            )
            return {
                "status": "success",
                "message": "LLM connection successful",
                "test_response": response.choices[0].message.content,
                "model": self.model,
            }
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            raise OpenAIError(
                message=f"Failed to connect to OpenAI API: {str(e)}",
                error_type="connection_error",
            )

    async def custom_query(
        self, user_query: str, max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """Send a custom query to GPT and get a response."""
        try:
            await self._check_rate_limit()

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_query},
                ],
                max_tokens=max_tokens,
            )

            response_text = response.choices[0].message.content
            if not response_text.strip():
                raise OpenAIError(
                    message="Empty response received from OpenAI",
                    error_type="empty_response",
                )

            return {
                "status": "success",
                "query": user_query,
                "response": response_text,
                "model": self.model,
            }
        except Exception as e:
            logger.error(f"Custom query failed: {str(e)}")
            raise OpenAIError(
                message=f"Failed to process query: {str(e)}", error_type="query_error"
            )

    def _validate_inputs(
        self, resume_content: str, job_description: str, preferences: Dict
    ) -> None:
        """Validate input parameters before processing."""
        if not resume_content or not resume_content.strip():
            raise ValueError("Resume content cannot be empty")

        if not job_description or not job_description.strip():
            raise ValueError("Job description cannot be empty")

        if not isinstance(preferences, dict):
            raise ValueError("Preferences must be a dictionary")

        # Check content length limits
        if len(resume_content) > 10000:
            raise ValueError("Resume content exceeds maximum length")

        if len(job_description) > 5000:
            raise ValueError("Job description exceeds maximum length")

    def _build_messages(
        self, resume_content: str, job_description: str, preferences: Dict
    ) -> List[Dict[str, str]]:
        """Build the message list for the GPT conversation."""
        system_prompt = """You are an expert resume enhancement AI that specializes in:
1. Optimizing resumes for ATS systems by incorporating relevant keywords naturally
2. Enhancing bullet points with specific, quantifiable achievements
3. Improving resume structure and formatting for better readability
4. Highlighting key skills and experiences that match job requirements
5. Maintaining professional language and tone throughout

Your task is to enhance resumes to match job descriptions while maintaining truthfulness and professionalism."""

        enhancement_prompt = f"""Please enhance this resume based on the following job description and preferences:

JOB DESCRIPTION:
{job_description}

CURRENT RESUME:
{resume_content}

ENHANCEMENT PREFERENCES:
- Preferred Job Titles: {preferences.get("preferred_job_titles", [])}
- Target Industries: {preferences.get("target_industries", [])}
- Keywords: {preferences.get("keywords", [])}

Please enhance the resume focusing on:
1. Natural keyword integration from the job description
2. Achievement quantification with metrics where possible
3. Bullet point enhancement with strong action verbs
4. ATS optimization
5. Skills highlighting matching job requirements

Return the response in this exact JSON format:
{{
    "enhanced_resume": "The complete enhanced resume text",
    "changes_summary": {{
        "keywords_added": ["list", "of", "keywords"],
        "formatting_updates": ["list", "of", "changes"],
        "skills_added": ["list", "of", "skills"],
        "bullets_rephrased": number_of_bullets_changed
    }}
}}"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": enhancement_prompt},
        ]

    async def enhance_resume(
        self, resume_content: str, job_description: str, preferences: Dict
    ) -> Dict[str, Any]:
        """
        Enhance a resume using GPT with improved error handling and debugging.

        This method attempts to get a JSON response from GPT, but can fall back to
        plain text if needed. It includes detailed logging to help diagnose issues.
        """
        try:
            # Validate inputs
            if not resume_content or not resume_content.strip():
                raise OpenAIError("Resume content cannot be empty", "validation_error")
            if not job_description or not job_description.strip():
                raise OpenAIError("Job description cannot be empty", "validation_error")

            # Building a more explicit system prompt
            system_prompt = """You are an expert resume enhancement AI. You must return your response in valid JSON format.
            Format your response EXACTLY like this, replacing the placeholders with actual content:
            {
                "enhanced_resume": "the complete enhanced resume text here",
                "changes_summary": {
                    "keywords_added": ["keyword1", "keyword2"],
                    "formatting_updates": ["change1", "change2"],
                    "skills_added": ["skill1", "skill2"],
                    "bullets_rephrased": 3
                }
            }"""

            # Building the user prompt
            user_prompt = f"""Please enhance this resume based on the job description.

            JOB DESCRIPTION:
            {job_description}

            CURRENT RESUME:
            {resume_content}

            PREFERENCES:
            - Preferred Titles: {preferences.get('preferred_job_titles', [])}
            - Target Industries: {preferences.get('target_industries', [])}
            - Keywords: {preferences.get('keywords', [])}

            Remember to return ONLY the JSON response in the exact format specified."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # Make the API call
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            # Get the raw response content
            raw_response = response.choices[0].message.content

            # Log the raw response for debugging
            logger.debug(
                f"Raw GPT response: {raw_response[:500]}..."
            )  # First 500 chars

            try:
                # Try to parse as JSON first
                enhancement_result = json.loads(raw_response)

                # Validate required fields
                if not all(
                    k in enhancement_result
                    for k in ["enhanced_resume", "changes_summary"]
                ):
                    raise OpenAIError(
                        "Response missing required fields",
                        "parsing_error",
                        {"raw_response": raw_response[:500]},
                    )

                return enhancement_result

            except json.JSONDecodeError:
                # If JSON parsing fails, provide a fallback response
                logger.warning(
                    "Failed to parse JSON response, falling back to plain text"
                )
                return {
                    "enhanced_resume": raw_response,
                    "changes_summary": {
                        "keywords_added": [],
                        "formatting_updates": ["Plain text response received"],
                        "skills_added": [],
                        "bullets_rephrased": 0,
                    },
                }

        except OpenAIError:
            raise
        except Exception as e:
            logger.error(f"Resume enhancement failed: {str(e)}")
            raise OpenAIError(
                f"Resume enhancement failed: {str(e)}", "processing_error"
            )

    async def _make_api_call_with_retry(
        self, messages: List[Dict[str, str]], max_retries: int = 3
    ) -> Any:
        """Make API calls to GPT with retry logic."""
        for attempt in range(max_retries):
            try:
                await self._check_rate_limit()

                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                return response

            except Exception as e:
                logger.warning(f"API call attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error("All retry attempts failed")
                    raise ValueError(
                        f"Failed to get response from GPT API after {max_retries} attempts: {str(e)}"
                    )
                await asyncio.sleep(1 * (attempt + 1))

    async def _check_rate_limit(self) -> None:
        """Ensure minimum time between API requests."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self._min_request_interval:
            await asyncio.sleep(self._min_request_interval - time_since_last)
        self._last_request_time = time.time()

    def _validate_enhancement_result(self, result: Dict) -> None:
        """Validate the enhancement result structure and content."""
        required_fields = ["enhanced_resume", "changes_summary"]
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")

        if (
            not isinstance(result["enhanced_resume"], str)
            or not result["enhanced_resume"].strip()
        ):
            raise ValueError("Enhanced resume content is empty or invalid")

        summary = result["changes_summary"]
        if not isinstance(summary, dict):
            raise ValueError("Changes summary must be a dictionary")

        required_summary_fields = {
            "keywords_added": list,
            "formatting_updates": list,
            "skills_added": list,
            "bullets_rephrased": (int, float),
        }

        for field, expected_type in required_summary_fields.items():
            if field not in summary:
                raise ValueError(f"Missing required summary field: {field}")
            if not isinstance(summary[field], expected_type):
                raise ValueError(
                    f"Invalid type for {field}: expected {expected_type}, got {type(summary[field])}"
                )
