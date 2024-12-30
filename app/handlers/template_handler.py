from typing import Dict, Any, Optional, List
import logging
import os
from pathlib import Path
from docxtpl import DocxTemplate
from fastapi import HTTPException, status
from fastapi.responses import FileResponse
from fastapi.background import BackgroundTasks
import tempfile
import io

logger = logging.getLogger(__name__)
# In template_handler.py


class DocxTemplateHandler:
    def __init__(self, template_path: Optional[str] = None):
        """
        Enhanced initialization with template validation.
        """
        try:
            # Get absolute path to template
            if template_path is None:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                template_path = os.path.join(
                    current_dir, "..", "templates", "ATS_Resume_Template.docx"
                )

            self.template_path = Path(template_path)
            logger.info(
                f"Initializing template handler with path: {self.template_path}"
            )

            # Verify template exists
            if not self.template_path.exists():
                raise FileNotFoundError(f"Template not found at: {self.template_path}")

            # Test load template and validate structure
            self._validate_template_structure()
            logger.info("Template initialization successful")

        except Exception as e:
            logger.error(f"Template initialization failed: {str(e)}")
            raise

    def _load_template(self) -> DocxTemplate:
        """
        Load the Word template file.

        Returns:
            DocxTemplate: The loaded template object

        Raises:
            ValueError: If there's an error loading the template
        """
        try:
            return DocxTemplate(str(self.template_path))
        except Exception as e:
            logger.error(f"Error loading template: {str(e)}")
            raise ValueError(f"Failed to load template: {str(e)}")

    def _validate_template_structure(self) -> None:
        """
        Validates the template structure and logs all placeholders.
        """
        try:
            doc = self._load_template()
            placeholders = set(doc.undeclared_template_variables)

            logger.debug("Template placeholders found:")
            for placeholder in sorted(placeholders):
                logger.debug(f"- {placeholder}")

            # Define expected fields
            expected_fields = {
                "candidate_name",
                "linkedin_username",
                "candidate_email",
                "candidate_phone",
                "job_title",
                "summary_text",
                "backend_skills",
                "frontend_skills",
                "database_skills",
                "cloud_technologies",
                "devops_skills",
                "methodologies",
                "soft_skills",
                "career_highlights",
                "career_highlight_1",
                "career_highlight_2",
                "career_highlight_3",
                "career_highlight_4",
                "professional_experience",
                "job1_company",
                "job1_location",
                "job1_title",
                "job1_dates",
                "job1_description",
                "key_responsibilities",
                "job1_responsibilities",
                "job2_company",
                "job2_location",
                "job2_title",
                "job2_dates",
                "job2_description",
                "job2_responsibilities",
                "job3_company",
                "job3_location",
                "job3_title",
                "job3_dates",
                "job3_description",
                "job3_responsibilities",
                "education_heading",
                "masters_title",
                "masters_institution",
                "masters_year",
                "bachelor_title",
                "bachelor_institution",
                "bachelor_year",
                # Add all your expected fields here
            }

            # Check for missing fields
            missing_fields = expected_fields - placeholders
            if missing_fields:
                logger.warning("Missing placeholders in template:")
                for field in missing_fields:
                    logger.warning(f"- {field}")

            # Check for unexpected fields
            extra_fields = placeholders - expected_fields
            if extra_fields:
                logger.warning("Unexpected placeholders in template:")
                for field in extra_fields:
                    logger.warning(f"- {field}")

        except Exception as e:
            logger.error(f"Template validation failed: {str(e)}")
            raise ValueError(f"Template validation failed: {str(e)}")

    async def generate_resume_docx(
        self, resume_data: Dict[str, Any], background_tasks: BackgroundTasks
    ) -> FileResponse:
        """
        Enhanced document generation with comprehensive logging and validation.
        """
        temp_file = None
        try:
            # Validate input data
            logger.debug("Validating resume data before generation...")
            missing_fields = self._validate_required_fields(resume_data)
            if missing_fields:
                raise ValueError(
                    f"Missing required fields: {', '.join(missing_fields)}"
                )

            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix=".docx", delete=False)
            logger.debug(f"Created temporary file: {temp_file.name}")

            # Load and render template
            doc = self._load_template()
            try:
                logger.debug("Rendering template...")
                doc.render(resume_data)
                logger.debug("Template rendered successfully")
            except Exception as e:
                logger.error(f"Template rendering failed: {str(e)}")
                raise ValueError(f"Failed to render template: {str(e)}")

            # Save document
            doc.save(temp_file.name)
            logger.debug("Document saved successfully")

            # Prepare response
            file_response = FileResponse(
                path=temp_file.name,
                filename="Enhanced_Resume.docx",
                media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )

            # Add cleanup task
            async def cleanup_temp_file():
                try:
                    os.unlink(temp_file.name)
                    logger.debug(f"Cleaned up temporary file: {temp_file.name}")
                except Exception as e:
                    logger.error(f"Failed to cleanup temp file: {str(e)}")

            background_tasks.add_task(cleanup_temp_file)

            logger.info("Document generation completed successfully")
            return file_response

        except Exception as e:
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except Exception as cleanup_error:
                    logger.error(f"Failed to cleanup temp file: {str(cleanup_error)}")

            logger.error("Document generation failed", exc_info=True)
            raise

    def _validate_required_fields(self, data: Dict[str, Any]) -> List[str]:
        """
        Validates that all required fields (as they appear in the template)
        are present and non-empty. Returns a list of missing field display
        names if any fields are missing or empty.
        """
        required_fields = [
            # 1. Basic Info
            ("candidate_name", "Candidate Name"),
            ("linkedin_username", "LinkedIn Username"),
            ("candidate_email", "Email"),
            ("candidate_phone", "Phone Number"),
            ("job_title", "Job Title"),
            ("summary_text", "Summary"),
            # 2. Key Skills
            ("key_skills", "Key Skills"),
            ("backend_skills", "Backend Skills"),
            ("frontend_skills", "Frontend Skills"),
            ("database_skills", "Database Skills"),
            ("cloud_technologies", "Cloud Technologies"),
            ("devops_skills", "DevOps & CI/CD"),
            ("methodologies", "Methodologies"),
            ("soft_skills", "Soft Skills"),
            # 3. Career Highlights
            ("career_highlights", "Career Highlights"),
            ("career_highlight_1", "Career Highlight 1"),
            ("career_highlight_2", "Career Highlight 2"),
            ("career_highlight_3", "Career Highlight 3"),
            ("career_highlight_4", "Career Highlight 4"),
            # 4. Professional Experience
            ("professional_experience", "Professional Experience"),
            # Job 1
            ("job1_company", "Job 1 Company"),
            ("job1_location", "Job 1 Location"),
            ("job1_title", "Job 1 Title"),
            ("job1_dates", "Job 1 Dates"),
            ("job1_description", "Job 1 Description"),
            ("job1_responsibilities", "Job 1 Responsibilities"),
            # Job 2
            ("job2_company", "Job 2 Company"),
            ("job2_location", "Job 2 Location"),
            ("job2_title", "Job 2 Title"),
            ("job2_dates", "Job 2 Dates"),
            ("job2_description", "Job 2 Description"),
            ("job2_responsibilities", "Job 2 Responsibilities"),
            # Job 3
            ("job3_company", "Job 3 Company"),
            ("job3_location", "Job 3 Location"),
            ("job3_title", "Job 3 Title"),
            ("job3_dates", "Job 3 Dates"),
            ("job3_description", "Job 3 Description"),
            ("job3_responsibilities", "Job 3 Responsibilities"),
            # 5. Education
            ("education_heading", "Education Heading"),
            ("masters_title", "Master's Title"),
            ("masters_institution", "Master's Institution"),
            ("masters_year", "Master's Year"),
            ("bachelor_title", "Bachelor's Title"),
            ("bachelor_institution", "Bachelor's Institution"),
            ("bachelor_year", "Bachelor's Year"),
        ]

        missing_fields = []

        for field_key, display_name in required_fields:
            # Check if the field is missing OR is empty/whitespace
            if (
                field_key not in data
                or not data[field_key]
                or not str(data[field_key]).strip()
            ):
                missing_fields.append(display_name)
                logger.warning(f"Missing required field: {display_name}")

        return missing_fields
