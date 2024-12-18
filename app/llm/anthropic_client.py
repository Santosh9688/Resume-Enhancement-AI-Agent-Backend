from anthropic import AsyncAnthropic
from typing import Any, Dict, List, Optional
import logging
import json
from datetime import datetime
import os
from dotenv import load_dotenv
import asyncio
import time

# Configure logging with more detailed format
logger = logging.getLogger(__name__)


class ResumeEnhancer:
    def __init__(self):
        """
        Initialize the ResumeEnhancer using environment variables for security.
        Sets up the Anthropic client and prepares conversation tracking.
        """
        # Load environment variables and validate API key
        load_dotenv()
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.error("ANTHROPIC_API_KEY not found in environment variables")
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")

        # Initialize Anthropic client with validation
        try:
            self.client = AsyncAnthropic(api_key=self.api_key)
            logger.info("Successfully initialized Anthropic client")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {str(e)}")
            raise

        # Initialize class variables
        self.conversation_history = []
        self.model = "claude-3-5-sonnet-20241022"
        self.max_tokens = 4096
        self.temperature = 0.7

        # [CHANGED] Merged the second __init__ logic here:
        # Rate limiting configuration
        self._last_request_time = 0
        self._min_request_interval = 0.1  # 100ms between requests

    async def test_connection(self) -> Dict[str, Any]:
        """Test the connection to Anthropic API."""
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=100,
                messages=[
                    {
                        "role": "user",
                        "content": "Hello, this is a test message. Please respond with 'Connection successful.'",
                    }
                ],
            )
            response_text = str(response.content)
            return {
                "status": "success",
                "message": "LLM connection successful",
                "test_response": response_text,
                "model": self.model,
            }
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            raise ValueError(f"Failed to connect to Anthropic API: {str(e)}")

    async def custom_query(
        self, user_query: str, max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Send a custom query to the LLM and get a response.

        This method allows you to send any question or prompt to the LLM and receive
        a structured response. It includes rate limiting and error handling for reliability.

        Args:
            user_query (str): The question or prompt you want to send to the LLM
            max_tokens (int): Maximum length of the response (default 1000)

        Returns:
            Dict containing:
            - status: Success or error status
            - query: The original query sent
            - response: The LLM's response
            - model: The model used for the response
        """
        try:
            # Apply rate limiting
            await self._check_rate_limit()

            # Make the API call with the user's query
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": user_query}],
            )
            response_text = str(response.content)
            if not response_text.strip():
                raise ValueError("Empty response received from LLM")
            return {
                "status": "success",
                "query": user_query,
                "response": response_text,
                "model": self.model,
            }
        except Exception as e:
            logger.error(f"Custom query failed: {str(e)}")
            raise ValueError(f"Failed to process query: {str(e)}")

    def _validate_inputs(
        self, resume_content: str, job_description: str, preferences: Dict
    ) -> None:
        """
        Validates input parameters before processing.
        """
        if not resume_content or not resume_content.strip():
            raise ValueError("Resume content cannot be empty")
        if not job_description or not job_description.strip():
            raise ValueError("Job description cannot be empty")
        if not isinstance(preferences, dict):
            raise ValueError("Preferences must be a dictionary")

        # Check resume and job description length limits
        if len(resume_content) > 10000:  # Arbitrary limit, adjust as needed
            raise ValueError("Resume content exceeds maximum length")
        if len(job_description) > 5000:  # Arbitrary limit, adjust as needed
            raise ValueError("Job description exceeds maximum length")

    def _build_system_prompt(self) -> str:
        """
        Builds the system prompt for Claude with resume enhancement instructions.
        """
        return """You are an expert resume enhancement AI that specializes in:
1. Optimizing resumes for ATS systems by incorporating relevant keywords naturally
2. Enhancing bullet points with specific, quantifiable achievements
3. Improving resume structure and formatting for better readability
4. Highlighting key skills and experiences that match job requirements
5. Maintaining professional language and tone throughout

For each enhancement request:
1. Analyze the job description to identify:
   - Required skills and qualifications
   - Key responsibilities
   - Industry-specific keywords
   - Company culture indicators

2. Review the resume to:
   - Match existing skills with job requirements
   - Identify areas for improvement
   - Note missing key qualifications
   - Evaluate current formatting

3. Make improvements by:
   - Adding missing relevant keywords naturally
   - Quantifying achievements where possible
   - Strengthening action verbs
   - Ensuring consistent formatting
   - Optimizing for ATS systems

4. Maintain:
   - Professional tone
   - Clear, concise language
   - ATS-friendly formatting
   - Original factual content"""

    def _build_enhancement_prompt(
        self, resume_content: str, job_description: str, preferences: Dict
    ) -> str:
        """
        Builds a detailed enhancement prompt that combines user inputs with instructions.
        """
        formatted_preferences = {
            "preferred_job_titles": preferences.get("preferred_job_titles", []),
            "target_industries": preferences.get("target_industries", []),
            "keywords": preferences.get("keywords", []),
        }

        return f"""Please enhance this resume based on the following job description and preferences:

JOB DESCRIPTION:
{job_description}

CURRENT RESUME:
{resume_content}

ENHANCEMENT PREFERENCES:
- Preferred Job Titles: {formatted_preferences["preferred_job_titles"]}
- Target Industries: {formatted_preferences["target_industries"]}
- Keywords: {formatted_preferences["keywords"]}

Please enhance the resume with these specific improvements:
1. Natural keyword integration:
   - Incorporate job description keywords
   - Add industry-specific terminology
   - Maintain natural language flow

2. Achievement quantification:
   - Add specific metrics where possible
   - Include percentages, numbers, and scales
   - Highlight concrete results

3. Bullet point enhancement:
   - Use strong action verbs
   - Focus on accomplishments
   - Maintain consistency in structure

4. ATS optimization:
   - Use standard section headings
   - Avoid complex formatting
   - Include key technical terms

5. Skills highlighting:
   - Emphasize relevant technical skills
   - Showcase transferable skills
   - Match job requirements

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

    # [CHANGED] Updated enhance_resume method to use retry logic and rate-limiting
    async def enhance_resume(
        self,
        resume_content: str,
        job_description: str,
        preferences: Dict,
    ) -> Dict[str, Any]:
        """
        Enhances a resume using Claude AI with improved reliability and validation.
        """
        try:
            # Validate inputs before processing
            self._validate_inputs(resume_content, job_description, preferences)

            logger.info("Starting resume enhancement process")
            logger.debug(
                f"Resume length: {len(resume_content)}, Job description length: {len(job_description)}"
            )

            # Build prompts
            system_prompt = self._build_system_prompt()
            enhancement_prompt = self._build_enhancement_prompt(
                resume_content, job_description, preferences
            )

            # [CHANGED] Rate limiting and API call with retry
            await self._check_rate_limit()
            response = await self._make_api_call_with_retry(
                system_prompt, enhancement_prompt
            )

            # Parse and validate response
            try:
                enhancement_result = json.loads(response.content)

                # Validate response structure
                self._validate_enhancement_result(enhancement_result)

                logger.info("Successfully processed and validated Claude API response")
                return enhancement_result

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Claude API response: {str(e)}")
                raise ValueError("Invalid JSON response from Claude API")

        except Exception as e:
            logger.error(f"Resume enhancement failed: {str(e)}", exc_info=True)
            raise

    async def _make_api_call_with_retry(
        self, system_prompt: str, enhancement_prompt: str, max_retries: int = 3
    ) -> Any:
        """
        Makes API calls to Claude with retry logic for better reliability.
        """
        for attempt in range(max_retries):
            try:
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": enhancement_prompt}],
                )
                return response

            except Exception as e:
                logger.warning(f"API call attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:  # Last attempt
                    logger.error("All retry attempts failed")
                    raise ValueError(
                        f"Failed to get response from Claude API after {max_retries} attempts: {str(e)}"
                    )
                await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff

    # [CHANGED] This method was added from the suggested changes
    async def _check_rate_limit(self):
        """
        Ensures minimum time between API requests.
        """
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self._min_request_interval:
            await asyncio.sleep(self._min_request_interval - time_since_last)
        self._last_request_time = time.time()

    def _validate_enhancement_result(self, result: Dict) -> None:
        """
        Performs comprehensive validation of the enhancement result.
        """
        # Validate top-level structure
        required_fields = ["enhanced_resume", "changes_summary"]
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")

        # Validate enhanced_resume content
        if (
            not isinstance(result["enhanced_resume"], str)
            or not result["enhanced_resume"].strip()
        ):
            raise ValueError("Enhanced resume content is empty or invalid")

        # Validate changes_summary structure
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

        # Validate list contents
        for field in ["keywords_added", "formatting_updates", "skills_added"]:
            if not all(isinstance(item, str) for item in summary[field]):
                raise ValueError(f"All items in {field} must be strings")
