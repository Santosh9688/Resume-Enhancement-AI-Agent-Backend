from typing import Any, Dict, List, Optional
import logging
import json
from datetime import datetime
import os
from dotenv import load_dotenv
import asyncio
import time
from openai import AsyncOpenAI
import aiohttp
from http import HTTPStatus
import re

# Configure logging
logger = logging.getLogger(__name__)


class DocxTemplateEnhancerError(Exception):
    """Custom exception class for DocxTemplateEnhancer-related errors."""

    def __init__(
        self,
        message: str,
        error_type: str = "enhancer_error",
        details: Optional[Dict] = None,
    ):
        self.message = message
        self.error_type = error_type
        self.details = details or {}
        super().__init__(self.message)


class DocxTemplateEnhancer:
    """
    A specialized class for enhancing resumes with AI and formatting them for docxtpl templates.
    This class specifically returns data structured to match Word template placeholders.
    """

    def __init__(self):
        """Initialize with better error handling and configuration."""
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("OPENAI_API_KEY not found in environment variables")
            raise DocxTemplateEnhancerError(
                message="OpenAI API key not found", error_type="configuration_error"
            )

        try:
            self.client = AsyncOpenAI(api_key=self.api_key)
            # Use the latest GPT-4 model with JSON mode
            self.model = "gpt-4o-2024-11-20"
            self.max_tokens = 4096
            self.temperature = 0.7
            logger.info("Successfully initialized OpenAI client")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise DocxTemplateEnhancerError(
                message=f"Failed to initialize OpenAI client: {str(e)}",
                error_type="initialization_error",
            )

    async def enhance_resume_for_template(
        self, resume_content: str, job_description: str, preferences: Dict
    ) -> Dict[str, Any]:
        """Enhanced resume processing with better validation and error handling."""
        try:
            # Validate inputs
            self._validate_inputs(resume_content, job_description, preferences)
            logger.debug("Input validation successful")

            # Build messages for the API call
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {
                    "role": "user",
                    "content": self._build_enhancement_prompt(
                        resume_content, job_description, preferences
                    ),
                },
            ]

            # Make API call with retry logic
            for attempt in range(3):  # Try up to 3 times
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        response_format={"type": "json_object"},  # Ensure JSON response
                    )
                    print("Raw LLM Response:", response.choices[0].message.content)
                    break
                except Exception as e:
                    if attempt == 2:  # Last attempt
                        raise
                    await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff

            # Process and validate the response
            enhancement_result = self._process_llm_response(
                response.choices[0].message.content
            )

            # Validate the structure
            if not self._validate_enhancement_structure(enhancement_result):
                logger.warning("Invalid response structure, using fallback")
                return self._create_fallback_response_with_original_content(
                    resume_content
                )

            return enhancement_result

        except Exception as e:
            logger.error(f"Resume enhancement failed: {str(e)}")
            return self._create_fallback_response_with_original_content(resume_content)

    def _get_system_prompt(self) -> str:
        """Return the system prompt for the LLM."""
        return """You are an expert resume enhancer that creates ATS-optimized resumes. 
        Your output must be valid JSON matching the template structure exactly.
        Key requirements:
        1. Extract accurate information from the resume
        2. Enhance content to match job requirements
        3. Use consistent date formats
        4. Create achievement-focused bullet points
        5. Maintain professional tone
        6. Return only valid JSON
        """

    def _validate_inputs(
        self, resume_content: str, job_description: str, preferences: Dict
    ) -> None:
        """Validate input parameters before processing."""
        if not resume_content or not resume_content.strip():
            raise DocxTemplateEnhancerError(
                "Resume content cannot be empty", "validation_error"
            )

        if not job_description or not job_description.strip():
            raise DocxTemplateEnhancerError(
                "Job description cannot be empty", "validation_error"
            )

        if not isinstance(preferences, dict):
            raise DocxTemplateEnhancerError(
                "Preferences must be a dictionary", "validation_error"
            )

        # Check content length limits
        if len(resume_content) > 10000:
            raise DocxTemplateEnhancerError(
                "Resume content exceeds maximum length of 10,000 characters",
                "validation_error",
            )

        if len(job_description) > 5000:
            raise DocxTemplateEnhancerError(
                "Job description exceeds maximum length of 5,000 characters",
                "validation_error",
            )

    def _build_enhancement_prompt(
        self, resume_content: str, job_description: str, preferences: Dict
    ) -> str:
        """Build a detailed prompt that ensures proper JSON output."""
        return f"""
        Enhance this resume for the job description and return as JSON matching this exact structure:
        {{
            "enhanced_resume_docx": {{
                "candidate_name": "string",
                "linkedin_username": "string",
                "candidate_email": "string",
                "candidate_phone": "string",
                "job_title": "string",
                "summary_text": "string",
                "key_skills": "KEY SKILLS",
                "backend_skills": "string",
                "frontend_skills": "string",
                "database_skills": "string",
                "cloud_technologies": "string",
                "devops_skills": "string",
                "methodologies": "string",
                "soft_skills": "string",
                "career_highlights": "CAREER HIGHLIGHTS",
                "career_highlight_1": "string",
                "career_highlight_2": "string",
                "career_highlight_3": "string",
                "career_highlight_4": "string",
                "professional_experience": "PROFESSIONAL EXPERIENCE",
                "job1_company": "string",
                "job1_location": "string",
                "job1_title": "string",
                "job1_dates": "string",
                "job1_description": "string",
                "job1_responsibilities": ["string"],
                "job2_company": "string",
                "job2_location": "string",
                "job2_title": "string",
                "job2_dates": "string",
                "job2_description": "string",
                "job2_responsibilities": ["string"],
                "job3_company": "string",
                "job3_location": "string",
                "job3_title": "string",
                "job3_dates": "string",
                "job3_description": "string",
                "job3_responsibilities": ["string"],
                "education_heading": "EDUCATION",
                "masters_title": "string",
                "masters_institution": "string",
                "masters_year": "string",
                "bachelor_title": "string",
                "bachelor_institution": "string",
                "bachelor_year": "string"
            }},
            "changes_summary": {{
                "keywords_added": ["string"],
                "formatting_updates": ["string"],
                "skills_added": ["string"],
                "bullets_rephrased": 0
            }}
        }}

        RESUME:
        {resume_content}

        JOB DESCRIPTION:
        {job_description}

        PREFERENCES:
        {json.dumps(preferences, indent=2)}
        """

    def _process_llm_response(self, raw_response: str) -> Dict[str, Any]:
        """Process LLM response with improved error handling."""
        try:
            # Parse JSON response
            enhancement_result = json.loads(raw_response)
            logger.debug("Successfully parsed JSON response")

            # Ensure all required fields exist
            if "enhanced_resume_docx" in enhancement_result:
                enhancement_result["enhanced_resume_docx"] = (
                    self._ensure_all_fields_exist(
                        enhancement_result["enhanced_resume_docx"]
                    )
                )

            return enhancement_result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
            raise DocxTemplateEnhancerError(
                message="Invalid JSON response from LLM",
                error_type="json_error",
                details={"raw_response": raw_response},
            )

    def _ensure_all_fields_exist(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all template fields exist with default values if needed."""
        required_fields = {
            "candidate_name": "",
            "linkedin_username": "",
            "candidate_email": "",
            "candidate_phone": "",
            "job_title": "",
            "summary_text": "",
            "key_skills": "KEY SKILLS",
            "backend_skills": "",
            "frontend_skills": "",
            "database_skills": "",
            "cloud_technologies": "",
            "devops_skills": "",
            "methodologies": "",
            "soft_skills": "",
            "career_highlights": "CAREER HIGHLIGHTS",
            "career_highlight_1": "",
            "career_highlight_2": "",
            "career_highlight_3": "",
            "career_highlight_4": "",
            "professional_experience": "PROFESSIONAL EXPERIENCE",
            "job1_company": "",
            "job1_location": "",
            "job1_title": "",
            "job1_dates": "",
            "job1_description": "",
            "job1_responsibilities": [],
            "job2_company": "",
            "job2_location": "",
            "job2_title": "",
            "job2_dates": "",
            "job2_description": "",
            "job2_responsibilities": [],
            "job3_company": "",
            "job3_location": "",
            "job3_title": "",
            "job3_dates": "",
            "job3_description": "",
            "job3_responsibilities": [],
            "education_heading": "EDUCATION",
            "masters_title": "",
            "masters_institution": "",
            "masters_year": "",
            "bachelor_title": "",
            "bachelor_institution": "",
            "bachelor_year": "",
        }

        # Update with existing values and ensure all fields exist
        result = required_fields.copy()
        result.update({k: v for k, v in data.items() if k in required_fields})

        return result

    def _extract_basic_info(self, text: str) -> Dict[str, str]:
        """Extract basic information using regex patterns."""
        patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "name": r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
            "linkedin": r"linkedin\.com/in/([a-zA-Z0-9-]+)",
        }

        info = {}
        for key, pattern in patterns.items():
            matches = re.findall(pattern, text, re.MULTILINE)
            info[key] = matches[0] if matches else ""

        return info

    def _create_fallback_response_with_original_content(
        self, original_content: str
    ) -> Dict[str, Any]:
        """Create a fallback response using the original content."""
        # Extract basic information
        basic_info = self._extract_basic_info(original_content)

        # Create base structure with extracted info
        fallback_data = {
            # Basic Info
            "candidate_name": basic_info.get("name", "Name Not Found"),
            "linkedin_username": basic_info.get("linkedin", "LinkedIn Not Found"),
            "candidate_email": basic_info.get("email", "Email Not Found"),
            "candidate_phone": basic_info.get("phone", "Phone Not Found"),
            "job_title": "Job Title Not Found",
            "summary_text": "Summary Not Found",
            # Section Headers
            "key_skills": "KEY SKILLS",
            "career_highlights": "CAREER HIGHLIGHTS",
            "professional_experience": "PROFESSIONAL EXPERIENCE",
            "education_heading": "EDUCATION",
            "key_responsibilities": "Key Responsibilities",
            # Skills Section
            "backend_skills": "Backend Skills Not Found",
            "frontend_skills": "Frontend Skills Not Found",
            "database_skills": "Database Skills Not Found",
            "cloud_technologies": "Cloud Technologies Not Found",
            "devops_skills": "DevOps & CI/CD Not Found",
            "methodologies": "Methodologies Not Found",
            "soft_skills": "Soft Skills Not Found",
            # Career Highlights
            "career_highlight_1": "Career Highlight 1 Not Found",
            "career_highlight_2": "Career Highlight 2 Not Found",
            "career_highlight_3": "Career Highlight 3 Not Found",
            "career_highlight_4": "Career Highlight 4 Not Found",
            # Professional Experience - Job 1
            "job1_company": "Company 1 Not Found",
            "job1_location": "Location 1 Not Found",
            "job1_title": "Job Title 1 Not Found",
            "job1_dates": "Dates 1 Not Found",
            "job1_description": "Description 1 Not Found",
            "job1_responsibilities": [],
            # Professional Experience - Job 2
            "job2_company": "Company 2 Not Found",
            "job2_location": "Location 2 Not Found",
            "job2_title": "Job Title 2 Not Found",
            "job2_dates": "Dates 2 Not Found",
            "job2_description": "Description 2 Not Found",
            "job2_responsibilities": [],
            # Professional Experience - Job 3
            "job3_company": "Company 3 Not Found",
            "job3_location": "Location 3 Not Found",
            "job3_title": "Job Title 3 Not Found",
            "job3_dates": "Dates 3 Not Found",
            "job3_description": "Description 3 Not Found",
            "job3_responsibilities": [],
            # Education Section
            "masters_title": "Master's Title Not Found",
            "masters_institution": "Master's Institution Not Found",
            "masters_year": "Master's Year Not Found",
            "bachelor_title": "Bachelor's Title Not Found",
            "bachelor_institution": "Bachelor's Institution Not Found",
            "bachelor_year": "Bachelor's Year Not Found",
        }

        # Ensure all required fields exist
        complete_data = self._ensure_all_fields_exist(fallback_data)

        return {
            "enhanced_resume_docx": complete_data,
            "changes_summary": {
                "keywords_added": [],
                "formatting_updates": ["Using fallback response"],
                "skills_added": [],
                "bullets_rephrased": 0,
            },
        }

    def _validate_enhancement_structure(self, result: Dict) -> bool:
        """Validate the enhancement result structure."""
        required_sections = {"enhanced_resume_docx", "changes_summary"}
        if not all(section in result for section in required_sections):
            return False

        # Validate changes_summary structure
        required_summary_fields = {
            "keywords_added": list,
            "formatting_updates": list,
            "skills_added": list,
            "bullets_rephrased": (int, float),
        }

        summary = result.get("changes_summary", {})
        if not all(
            field in summary and isinstance(summary[field], type_)
            for field, type_ in required_summary_fields.items()
        ):
            return False

        return True

    def _create_fallback_response(self) -> Dict[str, Any]:
        """Create a fallback response when JSON parsing fails."""
        return {
            "enhanced_resume_docx": {
                # All template fields with empty values
                "candidate_name": "",
                "linkedin_username": "",
                "candidate_email": "",
                "candidate_phone": "",
                # ... (all other fields)
            },
            "changes_summary": {
                "keywords_added": [],
                "formatting_updates": ["Failed to parse structured response"],
                "skills_added": [],
                "bullets_rephrased": 0,
            },
        }

    async def _check_rate_limit(self) -> None:
        """Ensure minimum time between API requests."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self._min_request_interval:
            await asyncio.sleep(self._min_request_interval - time_since_last)
        self._last_request_time = time.time()
