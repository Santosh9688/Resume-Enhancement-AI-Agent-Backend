from app.core.database import TransactionManager
from app.core.formatters import ResponseFormatter
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    status,
    File,
    UploadFile,
    Form,
    Path,
)
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from typing import Dict, Optional, Any, List
from datetime import datetime
from pydantic import BaseModel, Field
from app.validators.input_validator import EnhancementRequest
from app.validators.file_validator import FileValidator
from app.validators.resume_validator import ResumeValidator
from app.core.exceptions import ValidationError, FileProcessingError
import json
import logging
from app.database import get_db
from app.models.models import ResumeEnhancement, JobDescription, Resume, User
from app.llm.anthropic_client import ResumeEnhancer  # We'll create this next
import pypdf
import docx
import io
from tempfile import NamedTemporaryFile
import os
import magic

resume_enhancer = ResumeEnhancer()
# Configure enhanced logging
logger = logging.getLogger("file_processing")
# Add this at the start of your file for more detailed logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger("resume_enhancement")

# Constants for validation
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_TEXT_LENGTH = 50000  # Maximum characters for text fields
MIN_TEXT_LENGTH = 50  # Minimum characters for job description
ALLOWED_MIME_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
}


# Custom exceptions for better error handling
class InputValidationError(Exception):
    """Custom exception for input validation failures"""

    pass


class FileProcessingError(Exception):
    """Custom exception for file processing failures"""

    pass


class DatabaseError(Exception):
    """Custom exception for database operation failures"""

    pass


# Response models
class EnhancementResponse(BaseModel):
    """
    Defines the structure for enhancement responses.
    This ensures consistent API output format.
    """

    status: str = Field(
        ..., description="Status of the enhancement process", example="success"
    )
    enhancement_id: int = Field(
        ..., description="Unique identifier for this enhancement", example=1
    )
    enhanced_resume: str = Field(
        ...,
        description="The complete enhanced resume content",
        example="PROFESSIONAL SUMMARY\n\nExperienced software engineer...",
    )
    changes_summary: Dict = Field(
        default={
            "keywords_added": [],
            "formatting_updates": [],
            "skills_added": [],
            "bullets_rephrased": 0,
        },
        description="Detailed summary of changes made to the resume",
    )


class EnhancementDetails(BaseModel):
    """
    Defines the structure for retrieving enhancement details.
    Used by the get_enhancement endpoint.
    """

    enhancement_id: int = Field(
        ..., description="Unique identifier for the enhancement", example=1
    )
    enhanced_content: str = Field(
        ...,
        description="The enhanced resume content",
        example="PROFESSIONAL SUMMARY\n\nExperienced software engineer...",
    )
    changes_applied: Dict = Field(
        ...,
        description="Summary of changes applied to the resume",
        example={
            "keywords_added": ["Python", "AWS"],
            "formatting_updates": ["Improved spacing", "Standardized bullets"],
            "skills_added": ["Cloud Architecture"],
            "bullets_rephrased": 5,
        },
    )
    created_date: datetime = Field(
        ...,
        description="Timestamp when the enhancement was created",
        example="2024-12-15T10:30:00",
    )


class EnhancementInputValidator:
    """Handles validation of enhancement request inputs"""

    @staticmethod
    def validate_text_field(
        text: str,
        field_name: str,
        min_length: int = 0,
        max_length: int = MAX_TEXT_LENGTH,
    ):
        """Validates text fields for length and content"""
        if not text or not text.strip():
            raise InputValidationError(f"{field_name} cannot be empty")

        text_length = len(text)
        if text_length < min_length:
            raise InputValidationError(
                f"{field_name} must be at least {min_length} characters long"
            )
        if text_length > max_length:
            raise InputValidationError(
                f"{field_name} exceeds maximum length of {max_length} characters"
            )

    @staticmethod
    def validate_file_name(filename: str):
        """Validates file name format and length"""
        if not filename or len(filename) > 255:
            raise InputValidationError("Invalid file name length")

        # Check for invalid characters in filename
        invalid_chars = '<>:"/\\|?*'
        if any(char in filename for char in invalid_chars):
            raise InputValidationError("File name contains invalid characters")


class ErrorResponse(BaseModel):
    """Model for error responses"""

    status: str = "error"
    message: str
    error_type: str
    details: Optional[Dict] = None


class LLMTestResponse(BaseModel):
    """Response model for LLM test endpoint"""

    status: str = Field(..., description="Status of the test", example="success")
    message: str = Field(
        ..., description="Status message", example="LLM connection successful"
    )
    test_response: str = Field(
        ...,
        description="Response from the LLM",
        example="The benefits of a well-structured resume...",
    )
    model: str = Field(
        ..., description="Model used for the test", example="claude-3-5-sonnet-20241022"
    )


class CustomQueryRequest(BaseModel):
    """
    Request model for custom LLM queries.
    Defines the structure for sending custom queries to the LLM.
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question or prompt to send to the LLM",
        example="What are the key elements of a strong resume summary?",
    )
    max_tokens: Optional[int] = Field(
        1000, ge=1, le=4096, description="Maximum length of the response", example=1000
    )


class CustomQueryResponse(BaseModel):
    """
    Response model for custom LLM queries.
    Defines the structure of the LLM's response.
    """

    status: str = Field(..., description="Status of the query", example="success")
    query: str = Field(
        ...,
        description="Original query that was sent",
        example="What are the key elements of a strong resume summary?",
    )
    response: str = Field(
        ...,
        description="Response from the LLM",
        example="A strong resume summary should include...",
    )
    model: str = Field(
        ...,
        description="Model used for the response",
        example="claude-3-5-sonnet-20241022",
    )


async def validate_user(user_id: int, db: Session) -> User:
    """Validate user exists and return user object."""
    user = db.query(User).filter(User.UserID == user_id).first()
    if not user:
        logger.warning(f"Invalid user_id provided: {user_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )
    return user


async def validate_file_type(file: UploadFile) -> str:
    """Validate file type and return file extension."""
    if file.content_type not in ALLOWED_MIME_TYPES:
        logger.warning(f"Invalid file type: {file.content_type}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Please upload PDF or DOCX files only",
        )

    file_extension = file.filename.split(".")[-1].lower()
    expected_extension = ALLOWED_MIME_TYPES[file.content_type]

    if file_extension != expected_extension:
        logger.warning(f"Extension mismatch: {file_extension} vs {expected_extension}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File extension does not match content type",
        )
    return file_extension


async def read_file_content(file: UploadFile) -> bytes:
    """Read and validate file content."""
    content = bytearray()
    size = 0

    # Read file in chunks
    chunk_size = 1024 * 1024  # 1MB chunks
    while chunk := await file.read(chunk_size):
        size += len(chunk)
        if size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024 * 1024)}MB",
            )
        content.extend(chunk)

    if size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file uploaded"
        )

    return bytes(content)


# File processing functions
async def process_file_with_cleanup(file: UploadFile) -> str:
    """Process uploaded file with proper cleanup and memory management"""
    temp_file = None
    try:
        # Create temporary file for processing
        temp_file = NamedTemporaryFile(delete=False)

        # Read and write in chunks
        chunk_size = 1024 * 1024  # 1MB chunks
        size = 0

        while chunk := await file.read(chunk_size):
            size += len(chunk)
            if size > MAX_FILE_SIZE:
                raise FileProcessingError(
                    f"File exceeds maximum size of {MAX_FILE_SIZE // (1024 * 1024)}MB"
                )
            temp_file.write(chunk)

        temp_file.flush()

        if size == 0:
            raise FileProcessingError("Empty file uploaded")

        # Process based on file type
        file_extension = file.filename.split(".")[-1].lower()
        if file_extension == "pdf":
            result = await process_pdf_file(temp_file.name)
        else:
            result = await process_docx_file(temp_file.name)

        return result

    finally:
        if temp_file:
            temp_file.close()
            try:
                os.unlink(temp_file.name)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file: {str(e)}")


async def process_pdf_file(file_path: str) -> str:
    """Process PDF file with enhanced validation"""
    try:
        with open(file_path, "rb") as file:
            pdf_reader = pypdf.PdfReader(file)
            if len(pdf_reader.pages) == 0:
                raise FileProcessingError("PDF file contains no pages")

            text_content = []
            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text()
                if not page_text.strip():
                    logger.warning(f"Empty text on page {page_num}")
                text_content.append(page_text)

            final_text = "\n".join(text_content)
            if not final_text.strip():
                raise FileProcessingError("No readable text found in PDF")

            return final_text

    except Exception as e:
        logger.error(f"PDF processing error: {str(e)}")
        raise FileProcessingError(f"Failed to process PDF: {str(e)}")


async def process_docx_file(file_path: str) -> str:
    """Process DOCX file with validation"""
    try:
        doc = docx.Document(file_path)
        text_content = []

        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_content.append(paragraph.text)

        final_text = "\n".join(text_content)
        if not final_text.strip():
            raise FileProcessingError("No readable text found in DOCX")

        return final_text
    except Exception as e:
        logger.error(f"DOCX processing error: {str(e)}")
        raise FileProcessingError(f"Failed to process DOCX: {str(e)}")


async def create_database_records(
    db: Session,
    user_id: int,
    resume_text: str,
    job_description: str,
    preferred_job_titles: Optional[str],
    keywords: Optional[str],
) -> tuple[Resume, JobDescription]:
    """Create database records with transaction management."""
    try:
        # Create Resume record
        resume = Resume(UserID=user_id, ResumeContent=resume_text, Version=1)
        db.add(resume)
        await db.flush()

        # Create JobDescription record
        job_desc = JobDescription(
            UserID=user_id,
            JobTitle=preferred_job_titles.split(",")[0]
            if preferred_job_titles
            else "Not specified",
            CompanyName="Not specified",
            Description=job_description,
            Keywords=keywords,
        )
        db.add(job_desc)
        await db.flush()

        return resume, job_desc

    except SQLAlchemyError as e:
        await db.rollback()
        logger.error(f"Database error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error occurred during record creation",
        )


# Database operations
async def handle_database_transaction(db: Session, transaction_func, *args, **kwargs):
    """Handle database transactions with retry logic"""
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            async with db.begin() as transaction:
                result = await transaction_func(*args, **kwargs)
                await transaction.commit()
                return result

        except SQLAlchemyError as e:
            retry_count += 1
            if retry_count >= max_retries:
                logger.error(
                    f"Database transaction failed after {max_retries} attempts: {str(e)}"
                )
                raise DatabaseError(f"Database operation failed: {str(e)}")

            logger.warning(f"Transaction attempt {retry_count} failed, retrying...")
            await transaction.rollback()


# Router instance
router = APIRouter()


@router.post(
    "/enhance-resume",
    response_model=EnhancementResponse,
    summary="Enhance a resume using AI",
    description="Process and enhance a resume using AI, optimizing for ATS and job requirements",
)
async def enhance_resume(
    resume_file: UploadFile = File(...),
    job_description: str = Form(...),
    user_id: int = Form(...),
    preferred_job_titles: Optional[str] = Form(None),
    target_industries: Optional[str] = Form(None),
    keywords: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """Enhanced resume processing with comprehensive validation and error handling"""
    # Debugging point 1: Log input parameters
    logger.debug(f"""
    Starting resume enhancement with:
    - User ID: {user_id}
    - File name: {resume_file.filename}
    - Job description length: {len(job_description)}
    """)

    try:
        # Stage 1: Input Validation
        logger.debug(f"Starting enhance_resume with session id: {id(db)}")
        logger.debug(f"Initial transaction status: {db.in_transaction()}")
        logger.debug("Starting input validation")
        request_data = {
            "job_description": job_description,
            "user_id": user_id,
            "preferred_job_titles": preferred_job_titles,
            "target_industries": target_industries,
            "keywords": keywords,
        }

        # Debugging point 2: Validate request data
        logger.debug(f"Validating request data: {request_data}")
        try:
            # Create and validate the request
            validated_data = EnhancementRequest(**request_data).model_dump()
            logger.debug(
                f"Request validation successful. Validated data: {validated_data}"
            )

            # Now we can use the validated data for the rest of our operations
            job_description = validated_data["job_description"]
            user_id = validated_data["user_id"]
            preferred_job_titles = validated_data.get("preferred_job_titles")
            target_industries = validated_data.get("target_industries")
            keywords = validated_data.get("keywords")

        except ValidationError as e:
            logger.warning(f"Request validation failed: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ErrorResponse(
                    message=str(e), error_type="validation_error"
                ).model_dump(),
            )

        # Stage 2: File Validation
        logger.debug("Starting file validation")
        try:
            # Debugging point 3: Check file content
            file_content = await resume_file.read()
            logger.debug(f"File size: {len(file_content)} bytes")

            await FileValidator.validate_file(file_content, resume_file.filename)
            await resume_file.seek(0)  # Reset file pointer

            # Process and extract text
            resume_text = await process_file_with_cleanup(resume_file)

            # Debugging point 4: Check extracted text
            logger.debug(f"Extracted text length: {len(resume_text)}")

        except ValidationError as e:
            logger.warning(f"File validation failed: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ErrorResponse(
                    message=str(e), error_type="file_validation_error"
                ).model_dump(),
            )

        # Stage 3: Database Operations
        logger.debug("Starting database operations")
        try:
            # Create the transaction manager instance
            transaction_manager = TransactionManager(db)
            # Debugging point 5: Database operation start
            logger.debug(
                f"Created TransactionManager with session id: {id(transaction_manager.db)}"
            )
            # Ensure no transaction is in progress
            if db.in_transaction():
                logger.warning("Existing transaction found, rolling back")
                await db.rollback()
            resume, job_desc = await transaction_manager.execute_operation(
                create_database_records,
                "Failed to create database records",
                db=db,
                user_id=user_id,
                resume_text=resume_text,
                job_description=job_description,
                preferred_job_titles=preferred_job_titles,
                keywords=keywords,
            )
            logger.debug(
                f"Database operation completed - Resume ID: {resume.ResumeID if resume else 'None'}"
            )
            return EnhancementResponse(
                status="success",
                enhancement_id=resume.ResumeID,
                enhanced_resume="Sample enhanced resume",  # Replace with actual enhancement
                changes_summary={"sample": "changes"},  # Replace with actual changes
            )
        except DatabaseError as e:
            logger.error(f"Database operation failed: {str(e)}")

        # Stage 4: AI Enhancement
        logger.debug("Starting AI enhancement")
        try:
            # Debugging point 6: AI processing start
            enhancement_result = await resume_enhancer.enhance_resume(
                resume_content=resume_text,
                job_description=job_description,
                preferences={
                    "preferred_job_titles": preferred_job_titles.split(",")
                    if preferred_job_titles
                    else [],
                    "target_industries": target_industries.split(",")
                    if target_industries
                    else [],
                    "keywords": keywords.split(",") if keywords else [],
                },
            )

            # Debugging point 7: Check AI results
            logger.debug(
                f"AI enhancement completed, result length: {len(enhancement_result.get('enhanced_resume', ''))}"
            )

        except Exception as e:
            logger.error(f"AI enhancement failed: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    message="AI enhancement failed",
                    error_type="ai_error",
                    details={"error": str(e)},
                ).model_dump(),
            )

        # Stage 5: Final Response
        # Debugging point 8: Prepare successful response
        logger.debug("Preparing successful response")
        return EnhancementResponse(
            status="success",
            enhancement_id=resume.ResumeID,  # Use the actual ID
            enhanced_resume=enhancement_result["enhanced_resume"],
            changes_summary=enhancement_result["changes_summary"],
        )

    except Exception as e:
        # Debugging point 9: Catch unexpected errors
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                message="An unexpected error occurred",
                error_type="system_error",
                details={"error": str(e)},
            ).model_dump(),
        )


@router.get(
    "/{enhancement_id}",
    response_model=EnhancementDetails,
    summary="Retrieve enhancement details",
    description="Retrieves the complete details of a specific resume enhancement",
)
async def get_enhancement(
    enhancement_id: int = Path(
        ..., description="The ID of the enhancement to retrieve", example=1, gt=0
    ),
    db: Session = Depends(get_db),
):
    """Retrieves detailed information about a specific resume enhancement."""
    try:
        enhancement = (
            db.query(ResumeEnhancement)
            .filter(ResumeEnhancement.EnhancementID == enhancement_id)
            .first()
        )

        if not enhancement:
            logger.warning(f"Enhancement ID {enhancement_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Enhancement with ID {enhancement_id} not found",
            )

        logger.info(f"Successfully retrieved enhancement {enhancement_id}")

        return EnhancementDetails(
            enhancement_id=enhancement.EnhancementID,
            enhanced_content=enhancement.EnhancedContent,
            changes_applied=json.loads(enhancement.ChangesApplied),
            created_date=enhancement.CreatedDate,
        )

    except SQLAlchemyError as e:
        logger.error(
            f"Database error retrieving enhancement {enhancement_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error occurred while retrieving enhancement",
        )


@router.post(
    "/test-llm",
    response_model=LLMTestResponse,
    summary="Test Anthropic LLM Connection",
    description="Simple endpoint to test if the Anthropic LLM integration is working",
)
async def test_llm_connection():
    """
    Test endpoint to verify Anthropic LLM connection is working properly.
    Returns a simple response from the LLM to confirm connectivity.
    """
    try:
        # Initialize the ResumeEnhancer
        enhancer = ResumeEnhancer()

        # Test the connection using the test_connection method
        response = await enhancer.test_connection()
        formatted_response = {
            "status": str(response["status"]),
            "message": str(response["message"]),
            "test_response": str(response["test_response"]),
            "model": str(response["model"]),
        }
        # Return response in the correct format
        return LLMTestResponse(**formatted_response)

    except Exception as e:
        logger.error(f"LLM test failed: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                message="LLM connection test failed",
                error_type="llm_error",
                details={"error": str(e)},
            ).model_dump(),
        )


@router.post(
    "/custom-query",
    response_model=CustomQueryResponse,
    summary="Send a custom query to the LLM",
    description="Send any question or prompt to the LLM and get a detailed response",
)
async def custom_llm_query(request: CustomQueryRequest):
    """
    Send a custom query to the LLM and get a response.
    This endpoint allows you to ask any question or provide any prompt to the LLM.

    The LLM will process your query and return a detailed response based on its
    training and knowledge.
    """
    try:
        # Initialize the ResumeEnhancer
        enhancer = ResumeEnhancer()

        # Send the query and get response
        response = await enhancer.custom_query(
            user_query=request.query, max_tokens=request.max_tokens
        )

        # Format and validate the response
        formatted_response = {
            "status": str(response["status"]),
            "query": str(response["query"]),
            "response": str(response["response"]),
            "model": str(response["model"]),
        }

        # Return the formatted response
        return CustomQueryResponse(**formatted_response)

    except Exception as e:
        # Log and handle any errors
        logger.error(f"Custom query failed: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                message="Failed to process query",
                error_type="llm_error",
                details={"error": str(e)},
            ).model_dump(),
        )
