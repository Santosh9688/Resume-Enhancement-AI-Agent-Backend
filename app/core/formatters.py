from typing import Optional, Dict, Any
from datetime import datetime


class ResponseFormatter:
    """
    Provides consistent formatting for all API responses.
    Think of this as a template system for API responses.
    """

    @staticmethod
    def success(
        data: Dict[str, Any],
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Formats a successful response with consistent structure.

        Args:
            data: The main response data
            message: Optional success message
            metadata: Optional metadata about the operation

        Returns:
            A formatted success response
        """
        response = {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
        }

        if message:
            response["message"] = message

        if metadata:
            response["metadata"] = metadata

        return response

    @staticmethod
    def error(
        message: str,
        error_type: str,
        details: Optional[Dict[str, Any]] = None,
        code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Formats an error response with consistent structure.

        Args:
            message: The error message
            error_type: Type of error (e.g., "validation_error")
            details: Optional additional error details
            code: Optional error code for reference

        Returns:
            A formatted error response
        """
        response = {
            "status": "error",
            "timestamp": datetime.utcnow().isoformat(),
            "error": {"type": error_type, "message": message},
        }

        if details:
            response["error"]["details"] = details

        if code:
            response["error"]["code"] = code

        return response


class EnhancementResponseFormatter(ResponseFormatter):
    """
    Specialized formatter for resume enhancement responses.
    Inherits from base ResponseFormatter but adds enhancement-specific formatting.
    """

    @staticmethod
    def format_enhancement_result(
        enhancement_id: int,
        enhanced_resume: str,
        changes_summary: Dict[str, Any],
        processing_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Formats a resume enhancement result.

        Args:
            enhancement_id: ID of the enhancement record
            enhanced_resume: The enhanced resume content
            changes_summary: Summary of changes made
            processing_time: Optional processing time in seconds

        Returns:
            Formatted enhancement response
        """
        data = {
            "enhancement_id": enhancement_id,
            "enhanced_resume": enhanced_resume,
            "changes_summary": changes_summary,
        }

        metadata = {}
        if processing_time is not None:
            metadata["processing_time_seconds"] = processing_time

        return ResponseFormatter.success(
            data=data,
            message="Resume enhancement completed successfully",
            metadata=metadata,
        )


# Example usage:
# response = EnhancementResponseFormatter.format_enhancement_result(
#     enhancement_id=123,
#     enhanced_resume="Enhanced resume content...",
#     changes_summary={
#         "keywords_added": ["python", "fastapi"],
#         "bullets_improved": 5
#     },
#     processing_time=2.5
# )
