"""
This module centralizes all custom exceptions used throughout the application.
By keeping all exceptions in one place, we maintain consistency in error handling
and make it easier to manage different types of errors.
"""

from typing import Optional, Dict, Any
from fastapi import status


class ResumeEnhancementError(Exception):
    """
    Base exception class for all resume enhancement related errors.
    This provides a consistent structure for error handling across the application.
    """

    def __init__(
        self,
        message: str,
        error_type: str = "general_error",
        details: Optional[Dict[str, Any]] = None,
        status_code: int = status.HTTP_400_BAD_REQUEST,
    ):
        self.message = message
        self.error_type = error_type
        self.details = details or {}
        self.status_code = status_code
        super().__init__(self.message)


class ValidationError(ResumeEnhancementError):
    """
    Raised when any kind of validation fails (input, file, content).
    Used for all validation-related errors across the application.
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_type="validation_error",
            details=details,
            status_code=status.HTTP_400_BAD_REQUEST,
        )


class FileProcessingError(ResumeEnhancementError):
    """
    Specific exception for file-related errors.
    This includes file reading, writing, and format issues.
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_type="file_processing_error",
            details=details,
            status_code=status.HTTP_400_BAD_REQUEST,
        )


class DatabaseError(ResumeEnhancementError):
    """
    Raised when database operations fail.
    This helps distinguish database errors from other types of errors.
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_type="database_error",
            details=details,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


class AIEnhancementError(ResumeEnhancementError):
    """
    Specific to AI processing errors.
    Used when the AI enhancement process fails.
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_type="ai_enhancement_error",
            details=details,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
