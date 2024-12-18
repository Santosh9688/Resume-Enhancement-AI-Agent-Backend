from typing import List, Dict, Optional, Set
import re
from pathlib import Path
import magic  # for better file type detection
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Base class for validation errors"""

    pass


class ResumeContentValidator:
    """
    Validates resume content for completeness and quality.
    Think of this as a quality control system for resumes.
    """

    # Required sections that should be present in a resume
    REQUIRED_SECTIONS = {"experience", "education", "skills"}

    # Minimum length requirements for different sections
    SECTION_MIN_LENGTHS = {
        "experience": 100,  # characters
        "education": 50,
        "skills": 30,
    }

    @classmethod
    async def validate_content(cls, content: str) -> None:
        """
        Validates the overall resume content.

        Args:
            content: The resume content to validate

        Raises:
            ValidationError: If content doesn't meet requirements
        """
        if not content or not content.strip():
            raise ValidationError("Resume content cannot be empty")

        # Check for required sections
        content_lower = content.lower()
        missing_sections = [
            section for section in cls.REQUIRED_SECTIONS if section not in content_lower
        ]

        if missing_sections:
            raise ValidationError(
                f"Resume missing required sections: {', '.join(missing_sections)}"
            )

        # Validate section lengths
        for section, min_length in cls.SECTION_MIN_LENGTHS.items():
            if not cls._validate_section_length(content_lower, section, min_length):
                raise ValidationError(
                    f"Section '{section}' is too short. "
                    f"Minimum {min_length} characters required."
                )

    @staticmethod
    def _validate_section_length(content: str, section: str, min_length: int) -> bool:
        """Helper method to validate section length"""
        # Simple section detection - could be made more sophisticated
        section_match = re.search(
            f"{section}.*?(?=education|experience|skills|$)",
            content,
            re.IGNORECASE | re.DOTALL,
        )

        if not section_match:
            return False

        section_content = section_match.group(0)
        return len(section_content.strip()) >= min_length


class FileValidator:
    """
    Validates file uploads with enhanced security checks.
    Think of this as a security scanner for uploaded files.
    """

    # Maximum file sizes for different types (in bytes)
    MAX_SIZES = {
        "pdf": 10 * 1024 * 1024,  # 10MB
        "docx": 5 * 1024 * 1024,  # 5MB
    }

    # Allowed MIME types
    ALLOWED_MIME_TYPES = {
        "application/pdf": "pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    }

    @classmethod
    async def validate_file(cls, file_content: bytes, filename: str) -> None:
        """
        Validates an uploaded file.

        Args:
            file_content: The file's content
            filename: Original filename

        Raises:
            ValidationError: If file doesn't meet requirements
        """
        # Validate filename
        cls._validate_filename(filename)

        # Check file size
        file_size = len(file_content)
        file_ext = Path(filename).suffix[1:].lower()

        if file_ext not in cls.MAX_SIZES:
            raise ValidationError(f"Unsupported file type: {file_ext}")

        if file_size > cls.MAX_SIZES[file_ext]:
            raise ValidationError(
                f"File too large. Maximum size for {file_ext} is "
                f"{cls.MAX_SIZES[file_ext] // (1024 * 1024)}MB"
            )

        # Validate file type using libmagic
        mime_type = magic.from_buffer(file_content, mime=True)
        if mime_type not in cls.ALLOWED_MIME_TYPES:
            raise ValidationError(f"Invalid file type detected: {mime_type}")

    @staticmethod
    def _validate_filename(filename: str) -> None:
        """Validates filename for security"""
        if not filename or len(filename) > 255:
            raise ValidationError("Invalid filename length")

        # Check for invalid characters
        invalid_chars = '<>:"/\\|?*'
        if any(char in filename for char in invalid_chars):
            raise ValidationError("Filename contains invalid characters")


class JobDescriptionValidator:
    """
    Validates job description content.
    Ensures job descriptions contain necessary information.
    """

    MIN_LENGTH = 100
    REQUIRED_ELEMENTS = {"responsibilities", "requirements", "qualifications"}

    @classmethod
    async def validate(cls, description: str) -> None:
        """
        Validates a job description.

        Args:
            description: The job description to validate

        Raises:
            ValidationError: If description doesn't meet requirements
        """
        if not description or len(description.strip()) < cls.MIN_LENGTH:
            raise ValidationError(
                f"Job description must be at least {cls.MIN_LENGTH} characters"
            )

        # Check for required elements
        description_lower = description.lower()
        missing_elements = [
            element
            for element in cls.REQUIRED_ELEMENTS
            if element not in description_lower
        ]

        if missing_elements:
            raise ValidationError(
                f"Job description missing required elements: "
                f"{', '.join(missing_elements)}"
            )


# Example usage:
# async def validate_enhancement_request(
#     resume_file: bytes,
#     filename: str,
#     job_description: str
# ):
#     # Validate file
#     await FileValidator.validate_file(resume_file, filename)
#
#     # Extract and validate resume content
#     resume_content = extract_text_from_file(resume_file, filename)
#     await ResumeContentValidator.validate_content(resume_content)
#
#     # Validate job description
#     await JobDescriptionValidator.validate(job_description)
