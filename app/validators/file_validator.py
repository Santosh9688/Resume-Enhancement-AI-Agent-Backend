"""
This module handles all file-related validations including type checking,
security validation, and content extraction validation.
"""

from pathlib import Path
import mimetypes
from typing import Tuple
import logging
from ..core.exceptions import ValidationError, FileProcessingError

# Configure logging
logger = logging.getLogger(__name__)


class FileValidator:
    """
    Comprehensive file validation system that ensures files are safe and properly formatted.
    This class provides methods for validating files before processing them.
    """

    # Define constants for validation
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_MIME_TYPES = {
        "application/pdf": [".pdf"],
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [
            ".docx"
        ],
    }

    @classmethod
    async def validate_file(cls, file_content: bytes, filename: str) -> None:
        """
        Validates a file's size, type, and runs security checks.

        Args:
            file_content: The raw file content as bytes
            filename: Original filename with extension

        Raises:
            ValidationError: If any validation check fails
            FileProcessingError: If file processing fails
        """
        try:
            # Validate file size
            if len(file_content) > cls.MAX_FILE_SIZE:
                raise ValidationError(
                    f"File too large. Maximum size is {cls.MAX_FILE_SIZE // (1024 * 1024)}MB"
                )

            # Validate filename
            cls._validate_filename(filename)

            # Check file type
            mime_type, encoding = cls._get_file_type(filename)
            if mime_type not in cls.ALLOWED_MIME_TYPES:
                raise ValidationError(f"Unsupported file type: {mime_type}")

            # Validate extension
            file_ext = Path(filename).suffix.lower()
            if file_ext not in cls.ALLOWED_MIME_TYPES[mime_type]:
                raise ValidationError("File extension does not match expected type")

            # Perform security checks
            await cls._perform_security_checks(file_content)

            logger.info(f"File validation passed for {filename}")

        except ValidationError:
            # Re-raise validation errors as is
            raise
        except Exception as e:
            # Wrap other errors in FileProcessingError
            logger.error(f"File processing error: {str(e)}")
            raise FileProcessingError(f"Failed to process file: {str(e)}")

    @staticmethod
    def _validate_filename(filename: str) -> None:
        """
        Validates the filename for length and invalid characters.

        Args:
            filename: The name of the file to validate

        Raises:
            ValidationError: If filename is invalid
        """
        if not filename or len(filename) > 255:
            raise ValidationError("Invalid filename length")

        # Check for invalid characters
        invalid_chars = '<>:"/\\|?*'
        if any(char in filename for char in invalid_chars):
            raise ValidationError("Filename contains invalid characters")

    @staticmethod
    def _get_file_type(filename: str) -> Tuple[str, str]:
        """
        Determines file type using Python's mimetypes library.

        Args:
            filename: The name of the file with extension

        Returns:
            Tuple of (mime_type, encoding)

        Raises:
            ValidationError: If file type cannot be determined
        """
        mimetypes.init()
        mime_type, encoding = mimetypes.guess_type(filename)

        if not mime_type:
            raise ValidationError("Could not determine file type")

        return mime_type, encoding

    @staticmethod
    async def _perform_security_checks(file_content: bytes) -> None:
        """
        Performs security checks on file content to detect potential threats.

        Args:
            file_content: The raw file content as bytes

        Raises:
            ValidationError: If security checks fail
        """
        content_lower = file_content.lower()

        # Define patterns that might indicate malicious content
        dangerous_patterns = [
            b"<?php",  # PHP code
            b"<script",  # JavaScript
            b"data:text/html",  # Data URLs
            b"<%",  # ASP tags
        ]

        for pattern in dangerous_patterns:
            if pattern in content_lower:
                raise ValidationError(
                    "Potential security risk detected in file",
                    details={"risk_type": "malicious_content"},
                )
