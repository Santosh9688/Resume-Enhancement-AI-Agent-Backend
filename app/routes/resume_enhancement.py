from app.core.database import TransactionManager
from app.core.formatters import ResponseFormatter
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    status,
    BackgroundTasks,
    File,
    UploadFile,
    Form,
    Path,
)
from fastapi import Path as FastAPIPath
import zipfile
from fastapi.responses import JSONResponse, FileResponse
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from typing import Dict, Optional, Any, List, Union
from datetime import datetime
from pydantic import BaseModel, Field
from app.validators.input_validator import EnhancementRequest
from app.validators.file_validator import FileValidator
from app.validators.resume_validator import ResumeValidator
from app.core.exceptions import ValidationError, FileProcessingError
from app.utils.resume_generator import resume_generator
import json
import logging
from app.database import get_db
from app.models.models import ResumeEnhancement, JobDescription, Resume, User
from app.llm.anthropic_client import ResumeEnhancer  # We'll create this next
from app.llm.openai_client import OpenAIResumeEnhancer, OpenAIError
from app.handlers.template_handler import DocxTemplateHandler
from app.llm.docxtemplate_resume_enhancer import DocxTemplateEnhancer
import pypdf
import docx
import tempfile
import io
from docxtpl import DocxTemplate
from pathlib import Path
from tempfile import NamedTemporaryFile
import os
import magic
import shutil

resume_enhancer = ResumeEnhancer()
openai_enhancer = OpenAIResumeEnhancer()
template_handler = DocxTemplateHandler(
    template_path=r"C:\Users\15613\Desktop\12-12@Resume AI Agent\resume-ai-backend\app\templates\ATS_Resume_Template.docx"
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="resume_generation.log",  # This creates a log file
    filemode="a",  # Append mode, so we don't overwrite previous logs
)
logger = logging.getLogger("resume_enhancement")
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


class ResumeDocxRequest(BaseModel):
    """
    Request model for DOCX resume generation that exactly matches the template structure.
    Each field corresponds to a placeholder in the DOCX template.
    """

    # Personal Information
    candidate_name: str = Field(..., description="Full name of the candidate")
    linkedin_username: str = Field(..., description="LinkedIn profile username")
    candidate_email: str = Field(..., description="Professional email address")
    candidate_phone: str = Field(..., description="Contact phone number")

    # Professional Summary
    job_title: str = Field(..., description="Current or desired job title")
    summary_text: str = Field(
        ..., description="Professional summary or objective statement"
    )

    # Section Headers (for customization)
    key_skills: str = Field(
        default="KEY SKILLS", description="Header for skills section"
    )
    career_highlights: str = Field(
        default="CAREER HIGHLIGHTS", description="Header for career highlights section"
    )
    professional_experience: str = Field(
        default="PROFESSIONAL EXPERIENCE", description="Header for experience section"
    )
    education_heading: str = Field(
        default="EDUCATION", description="Header for education section"
    )
    key_responsibilities: str = Field(
        default="Key Responsibilities",
        description="Header for responsibilities subsection",
    )

    # Skills Section
    backend_skills: Optional[str] = Field(
        None, description="Backend development skills"
    )
    frontend_skills: Optional[str] = Field(
        None, description="Frontend development skills"
    )
    database_skills: Optional[str] = Field(
        None, description="Database management skills"
    )
    cloud_technologies: Optional[str] = Field(
        None, description="Cloud technology skills"
    )
    devops_skills: Optional[str] = Field(None, description="DevOps and CI/CD skills")
    methodologies: Optional[str] = Field(None, description="Development methodologies")
    soft_skills: Optional[str] = Field(None, description="Soft skills and competencies")

    # Career Highlights
    career_highlight_1: Optional[str] = Field(
        None, description="First career highlight"
    )
    career_highlight_2: Optional[str] = Field(
        None, description="Second career highlight"
    )
    career_highlight_3: Optional[str] = Field(
        None, description="Third career highlight"
    )
    career_highlight_4: Optional[str] = Field(
        None, description="Fourth career highlight"
    )

    # Professional Experience - Job 1
    job1_company: Optional[str] = Field(None, description="First company name")
    job1_location: Optional[str] = Field(None, description="First job location")
    job1_title: Optional[str] = Field(None, description="First job title")
    job1_dates: Optional[str] = Field(None, description="First job duration")
    job1_description: Optional[str] = Field(None, description="First job description")
    job1_responsibilities: Optional[List[str]] = Field(
        None, description="First job responsibilities"
    )

    # Professional Experience - Job 2
    job2_company: Optional[str] = Field(None, description="Second company name")
    job2_location: Optional[str] = Field(None, description="Second job location")
    job2_title: Optional[str] = Field(None, description="Second job title")
    job2_dates: Optional[str] = Field(None, description="Second job duration")
    job2_description: Optional[str] = Field(None, description="Second job description")
    job2_responsibilities: Optional[List[str]] = Field(
        None, description="Second job responsibilities"
    )

    # Professional Experience - Job 3
    job3_company: Optional[str] = Field(None, description="Third company name")
    job3_location: Optional[str] = Field(None, description="Third job location")
    job3_title: Optional[str] = Field(None, description="Third job title")
    job3_dates: Optional[str] = Field(None, description="Third job duration")
    job3_description: Optional[str] = Field(None, description="Third job description")
    job3_responsibilities: Optional[List[str]] = Field(
        None, description="Third job responsibilities"
    )

    # Education Section
    masters_title: Optional[str] = Field(None, description="Master's degree title")
    masters_institution: Optional[str] = Field(None, description="Master's institution")
    masters_year: Optional[str] = Field(None, description="Master's completion year")
    bachelor_title: Optional[str] = Field(None, description="Bachelor's degree title")
    bachelor_institution: Optional[str] = Field(
        None, description="Bachelor's institution"
    )
    bachelor_year: Optional[str] = Field(None, description="Bachelor's completion year")


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
                    # Add a statement to avoid a syntax error
                    # for example, skip empty pages
                    continue
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
        db.flush()

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
    async with db.begin() as transaction:
        try:
            result = await transaction_func(*args, **kwargs)
            return result
        except SQLAlchemyError as e:
            await transaction.rollback()
            logger.error(f"Database error: {str(e)}")
            raise DatabaseError(f"Database operation failed: {str(e)}")


# Open AI Models
class OpenAITestResponse(BaseModel):
    """Response model for OpenAI test endpoint"""

    status: str = Field(..., description="Status of the test", example="success")
    message: str = Field(
        ..., description="Status message", example="OpenAI connection successful"
    )
    test_response: str = Field(
        ...,
        description="Response from OpenAI",
        example="This is a test response from GPT-4",
    )
    model: str = Field(
        ..., description="Model used for the test", example="gpt-4-turbo-preview"
    )


class OpenAICustomQueryRequest(BaseModel):
    """Request model for OpenAI custom queries"""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question or prompt to send to OpenAI",
        example="What are the key elements of a strong resume?",
    )
    max_tokens: Optional[int] = Field(
        1000, ge=1, le=4096, description="Maximum length of the response", example=1000
    )


class OpenAICustomQueryResponse(BaseModel):
    """Response model for OpenAI custom queries"""

    status: str = Field(..., description="Status of the query", example="success")
    query: str = Field(..., description="Original query that was sent")
    response: str = Field(..., description="Response from OpenAI")
    model: str = Field(..., description="Model used for the response")


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
            logger.debug(
                f"Created TransactionManager with session id: {id(transaction_manager.db)}"
            )

            # Execute the database operation and await the result
            resume, job_desc = transaction_manager.execute_operation(
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

        except DatabaseError as e:
            logger.error(f"Database operation failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
            )

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

            # Stage 5: Final Response
            logger.debug("Preparing successful response")
            return EnhancementResponse(
                status="success",
                enhancement_id=resume.ResumeID,
                enhanced_resume=enhancement_result["enhanced_resume"],
                changes_summary=enhancement_result["changes_summary"],
            )

        except Exception as e:
            logger.error(f"AI enhancement failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"AI enhancement failed: {str(e)}",
            )

    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
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
    enhancement_id: int = FastAPIPath(
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


@router.post(
    "/test-openai",
    response_model=OpenAITestResponse,
    summary="Test OpenAI Connection",
    description="Test endpoint to verify if the OpenAI integration is working properly",
)
async def test_openai_connection():
    """
    Tests the OpenAI connection and verifies that the integration is working.
    This endpoint serves as a health check for the OpenAI integration by:
    1. Verifying the API key is valid
    2. Testing the connection to OpenAI's servers
    3. Ensuring the model responds appropriately
    4. Validating response formatting
    """
    try:
        logger.info("Starting OpenAI connection test")

        # Attempt to make a test connection
        response = await openai_enhancer.test_connection()

        # Format the response according to our model
        formatted_response = OpenAITestResponse(
            status=response["status"],
            message=response["message"],
            test_response=response["test_response"],
            model=response["model"],
        )

        logger.info("OpenAI connection test completed successfully")
        return formatted_response

    except OpenAIError as e:
        # Handle OpenAI-specific errors
        logger.error(
            f"OpenAI test failed: {str(e)}",
            extra={"error_type": e.error_type, "details": e.details},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": str(e),
                "error_type": e.error_type,
                "details": e.details,
            },
        )

    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error during OpenAI test: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during the OpenAI connection test",
        )


@router.post(
    "/openai-query",
    response_model=OpenAICustomQueryResponse,
    summary="Send a custom query to OpenAI",
    description="Send any question or prompt to OpenAI and get a detailed response",
)
async def openai_custom_query(request: OpenAICustomQueryRequest):
    """
    Processes a custom query using OpenAI's API.
    This endpoint allows testing different prompts and receiving detailed responses.

    The endpoint performs several steps:
    1. Validates the input query and parameters
    2. Sends the request to OpenAI
    3. Processes and formats the response
    4. Handles any errors that occur
    """
    try:
        logger.info(
            "Processing custom OpenAI query",
            extra={
                "query_preview": request.query[:100],
                "max_tokens": request.max_tokens,
            },
        )

        # Send the query to OpenAI
        response = await openai_enhancer.custom_query(
            user_query=request.query, max_tokens=request.max_tokens
        )

        # Format the response according to our model
        formatted_response = OpenAICustomQueryResponse(
            status=response["status"],
            query=response["query"],
            response=response["response"],
            model=response["model"],
        )

        logger.info("Custom query processed successfully")
        return formatted_response

    except OpenAIError as e:
        # Handle OpenAI-specific errors with detailed logging
        logger.error(
            f"OpenAI query failed: {str(e)}",
            extra={
                "error_type": e.error_type,
                "details": e.details,
                "query_preview": request.query[:100],
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": str(e),
                "error_type": e.error_type,
                "details": e.details,
            },
        )

    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error in custom query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your query",
        )


@router.post(
    "/test-openai-enhancement",
    response_model=EnhancementResponse,
    summary="Test OpenAI Resume Enhancement",
    description="Test the resume enhancement functionality using OpenAI",
)
async def test_openai_enhancement(
    resume_text: str = Form(...),
    job_description: str = Form(...),
    preferred_titles: Optional[str] = Form(None),
    target_industries: Optional[str] = Form(None),
    keywords: Optional[str] = Form(None),
):
    """
    Tests the resume enhancement functionality with OpenAI.
    This endpoint allows testing the enhancement process without requiring
    file uploads or database operations.

    The process includes:
    1. Input validation and preprocessing
    2. Formatting preferences and requirements
    3. Sending the enhancement request to OpenAI
    4. Processing and returning the enhanced resume
    """
    try:
        logger.info(
            "Starting resume enhancement test",
            extra={
                "resume_length": len(resume_text),
                "job_desc_length": len(job_description),
                "has_preferences": bool(
                    preferred_titles or target_industries or keywords
                ),
            },
        )

        # Validate input lengths
        if len(resume_text.strip()) < 50:
            raise ValueError(
                "Resume text is too short - please provide at least 50 characters"
            )
        if len(job_description.strip()) < 50:
            raise ValueError(
                "Job description is too short - please provide at least 50 characters"
            )

        # Prepare preferences dictionary
        preferences = {
            "preferred_job_titles": preferred_titles.split(",")
            if preferred_titles
            else [],
            "target_industries": target_industries.split(",")
            if target_industries
            else [],
            "keywords": keywords.split(",") if keywords else [],
        }

        # Process enhancement request
        enhancement_result = await openai_enhancer.enhance_resume(
            resume_content=resume_text,
            job_description=job_description,
            preferences=preferences,
        )

        # Format the response using our response model
        formatted_response = EnhancementResponse(
            status="success",
            enhancement_id=0,  # Test endpoint doesn't create database records
            enhanced_resume=enhancement_result["enhanced_resume"],
            changes_summary=enhancement_result["changes_summary"],
        )

        logger.info("Resume enhancement test completed successfully")
        return formatted_response

    except OpenAIError as e:
        # Handle OpenAI-specific errors
        logger.error(
            f"OpenAI enhancement failed: {str(e)}",
            extra={"error_type": e.error_type, "details": e.details},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": str(e),
                "error_type": e.error_type,
                "details": e.details,
            },
        )

    except ValueError as ve:
        # Handle validation errors
        logger.warning(f"Validation error: {str(ve)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))

    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error in enhancement test: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during the enhancement process",
        )


def create_temp_directory() -> str:
    """Create a temporary directory with proper permissions and cleanup handling."""
    try:
        temp_dir = tempfile.mkdtemp(prefix="resume_gen_")
        os.chmod(temp_dir, 0o755)  # Read/write for owner, read/execute for others
        if not os.path.exists(temp_dir) or not os.access(temp_dir, os.W_OK):
            raise FileProcessingError("Failed to create writable temporary directory")
        return temp_dir
    except Exception as e:
        logger.error(f"Failed to create temporary directory: {str(e)}")
        raise FileProcessingError(f"Failed to create temporary directory: {str(e)}")


class CleanupTask:
    """Helper class for cleaning up temporary files."""

    def __init__(self, temp_dir: str):
        self.temp_dir = temp_dir

    async def cleanup(self):
        """Cleanup temporary directory with proper error handling."""
        try:
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(
                    f"Successfully cleaned up temporary directory: {self.temp_dir}"
                )
        except Exception as e:
            logger.error(
                f"Failed to cleanup temporary directory {self.temp_dir}: {str(e)}"
            )


def validate_enhancement_result(result: Dict[str, Any]) -> None:
    """Validate the enhancement result structure."""
    required_fields = ["enhanced_resume", "changes_summary"]
    if not isinstance(result, dict):
        raise ValueError("Enhancement result must be a dictionary")
    for field in required_fields:
        if field not in result:
            raise ValueError(f"Missing required field in enhancement result: {field}")
    if (
        not isinstance(result["enhanced_resume"], str)
        or not result["enhanced_resume"].strip()
    ):
        raise ValueError("Enhanced resume must be a non-empty string")
    if not isinstance(result["changes_summary"], dict):
        raise ValueError("Changes summary must be a dictionary")


# Add this to your resume_enhancement.py file, inside the generate_standard_resume function
@router.post(
    "/generate-resume-docx",
    response_class=FileResponse,
    summary="Generate a formatted resume DOCX",
    description="Generate a professionally formatted resume in DOCX format using provided data",
)
async def generate_resume_docx(
    request: ResumeDocxRequest, background_tasks: BackgroundTasks
):
    """
    Generate a formatted resume in DOCX format using the provided data.

    This endpoint:
    1. Validates the input data
    2. Generates a document using the template
    3. Returns the document as a downloadable file
    """
    try:
        logger.info(f"Starting resume generation for {request.candidate_name}")

        # Convert Pydantic model to dict
        resume_data = request.model_dump()

        # Generate and return the document
        return await resume_generator.generate_resume_docx(
            resume_data=resume_data, background_tasks=background_tasks
        )

    except Exception as e:
        logger.error(f"Failed to generate resume: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate resume: {str(e)}",
        )


@router.post(
    "/auto-generate-resume",
    response_class=FileResponse,
    summary="Automatically generate an enhanced resume DOCX",
    description="Process a resume with OpenAI and generate a formatted DOCX file",
)
async def auto_generate_resume(
    resume_file: UploadFile = File(...),
    job_description: str = Form(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    preferred_titles: Optional[str] = Form(None),
    target_industries: Optional[str] = Form(None),
    keywords: Optional[str] = Form(None),
):
    """
    This endpoint combines LLM enhancement with DOCX generation:
    1. Extracts text from the uploaded resume
    2. Processes it through OpenAI to enhance and structure it
    3. Generates a formatted DOCX using our template
    """
    try:
        # Extract resume text
        resume_text = await process_file_with_cleanup(resume_file)

        # Process through OpenAI
        enhancement_result = await openai_enhancer.enhance_resume(
            resume_content=resume_text,
            job_description=job_description,
            preferences={
                "preferred_job_titles": preferred_titles.split(",")
                if preferred_titles
                else [],
                "target_industries": target_industries.split(",")
                if target_industries
                else [],
                "keywords": keywords.split(",") if keywords else [],
            },
        )

        # Format the enhanced content into template structure
        template_data = format_enhanced_content_for_template(enhancement_result)

        # Generate DOCX
        return await resume_generator.generate_resume_docx(
            resume_data=template_data, background_tasks=background_tasks
        )

    except Exception as e:
        logger.error(f"Auto-generation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate enhanced resume: {str(e)}",
        )


def format_enhanced_content_for_template(enhancement_result: dict) -> dict:
    """
    Converts the LLM enhancement result into our template format.
    """
    # Parse the enhanced resume content
    enhanced_content = enhancement_result["enhanced_resume"]

    try:
        # Try to parse as JSON first (in case LLM returned structured data)
        structured_data = json.loads(enhanced_content)
    except json.JSONDecodeError:
        # If not JSON, parse the text content
        structured_data = parse_resume_text(enhanced_content)

    return structured_data


def parse_resume_text(text: str) -> dict:
    """
    Parses plain text resume into structured format matching our template.
    Uses basic text analysis to identify sections and content.
    """
    # Initialize template structure
    template_data = {
        "candidate_name": "",
        "linkedin_username": "",
        "candidate_email": "",
        "candidate_phone": "",
        "job_title": "",
        "summary_text": "",
        "backend_skills": "",
        "frontend_skills": "",
        "database_skills": "",
        "cloud_technologies": "",
        "devops_skills": "",
        "methodologies": "",
        "soft_skills": "",
        "career_highlights": [],
        "job1_responsibilities": [],
        "job2_responsibilities": [],
        "job3_responsibilities": [],
        "education_heading": "EDUCATION",
        "masters_title": "",
        "masters_institution": "",
        "masters_year": "",
        "bachelor_title": "",
        "bachelor_institution": "",
        "bachelor_year": "",
    }

    # Split into sections
    sections = text.split("\n\n")

    for section in sections:
        section = section.strip()

        # Basic section identification logic
        if section.lower().startswith("summary"):
            template_data["summary_text"] = section.split(":", 1)[1].strip()

        elif "experience" in section.lower():
            # Parse job details
            jobs = parse_job_experience(section)
            for idx, job in enumerate(jobs, 1):
                if idx <= 3:  # We support up to 3 jobs
                    prefix = f"job{idx}_"
                    template_data[f"{prefix}company"] = job.get("company", "")
                    template_data[f"{prefix}title"] = job.get("title", "")
                    template_data[f"{prefix}dates"] = job.get("dates", "")
                    template_data[f"{prefix}description"] = job.get("description", "")
                    template_data[f"{prefix}responsibilities"] = job.get(
                        "responsibilities", []
                    )

        elif "skills" in section.lower():
            skills = parse_skills_section(section)
            template_data.update(skills)

        elif "education" in section.lower():
            education = parse_education_section(section)
            template_data.update(education)

    return template_data


def parse_job_experience(text: str) -> list:
    """Parse job experience section into structured format."""
    jobs = []
    current_job = {}
    for line in text.split("\n"):
        line = line.strip()
        if line:
            # Basic job parsing logic
            if ":" in line and not line.startswith("-"):
                current_job = {"responsibilities": []}
                company_title = line.split(":")
                current_job["company"] = company_title[0].strip()
                current_job["title"] = (
                    company_title[1].strip() if len(company_title) > 1 else ""
                )
                jobs.append(current_job)
            elif line.startswith("-") or line.startswith("•"):
                if current_job is not None:
                    current_job["responsibilities"].append(line.lstrip("- •").strip())
            else:
                if current_job is not None:
                    if "description" not in current_job:
                        current_job["description"] = line
                    else:
                        current_job["description"] += " " + line

    return jobs


def parse_skills_section(text: str) -> dict:
    """Parse skills section into structured format."""
    skills = {
        "backend_skills": "",
        "frontend_skills": "",
        "database_skills": "",
        "cloud_technologies": "",
        "devops_skills": "",
        "methodologies": "",
        "soft_skills": "",
    }

    current_category = None
    for line in text.split("\n"):
        line = line.strip()
        if line:
            if ":" in line:
                category, content = line.split(":", 1)
                category = category.strip().lower()
                if "backend" in category:
                    current_category = "backend_skills"
                elif "frontend" in category:
                    current_category = "frontend_skills"
                elif "database" in category:
                    current_category = "database_skills"
                elif "cloud" in category:
                    current_category = "cloud_technologies"
                elif "devops" in category:
                    current_category = "devops_skills"
                elif "methodologies" in category:
                    current_category = "methodologies"
                elif "soft" in category:
                    current_category = "soft_skills"

                if current_category:
                    skills[current_category] = content.strip()
            elif current_category and line:
                skills[current_category] += " " + line

    return skills


def parse_education_section(text: str) -> dict:
    """Parse education section into structured format."""
    education = {
        "masters_title": "",
        "masters_institution": "",
        "masters_year": "",
        "bachelor_title": "",
        "bachelor_institution": "",
        "bachelor_year": "",
    }

    for line in text.split("\n"):
        line = line.strip()
        if line:
            if "master" in line.lower():
                parts = line.split("|")
                if len(parts) >= 3:
                    education["masters_title"] = parts[0].strip()
                    education["masters_institution"] = parts[1].strip()
                    education["masters_year"] = parts[2].strip()
            elif "bachelor" in line.lower():
                parts = line.split("|")
                if len(parts) >= 3:
                    education["bachelor_title"] = parts[0].strip()
                    education["bachelor_institution"] = parts[1].strip()
                    education["bachelor_year"] = parts[2].strip()

    return education


@router.post("/auto-generate-resume-docx")
async def auto_generate_resume_docx(
    resume_file: UploadFile = File(...),
    job_description: str = Form(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    preferred_titles: Optional[str] = Form(None),
    target_industries: Optional[str] = Form(None),
    keywords: Optional[str] = Form(None),
):
    """
    Enhanced endpoint for resume generation with comprehensive logging and validation.
    """
    try:
        # Step 1: Log initial request
        logger.info("Starting resume generation process")
        logger.debug(f"""
        Processing request with:
        - File name: {resume_file.filename}
        - Job description length: {len(job_description)}
        - Preferred titles: {preferred_titles}
        - Target industries: {target_industries}
        - Keywords: {keywords}
        """)

        # Step 2: Validate file
        if not resume_file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="No file provided"
            )

        # Step 3: Process file
        try:
            resume_text = await process_file_with_cleanup(resume_file)
            logger.debug(
                f"Successfully extracted {len(resume_text)} characters from resume"
            )

            # Step 4: Initialize enhancer and process resume
            try:
                docx_enhancer = DocxTemplateEnhancer()
                logger.debug("Created DocxTemplateEnhancer instance")

                # Prepare preferences dictionary
                preferences = {
                    "preferred_job_titles": preferred_titles.split(",")
                    if preferred_titles
                    else [],
                    "target_industries": target_industries.split(",")
                    if target_industries
                    else [],
                    "keywords": keywords.split(",") if keywords else [],
                }
                logger.debug(
                    f"Prepared preferences: {json.dumps(preferences, indent=2)}"
                )

                # Process through enhancer
                enhancement_result = await docx_enhancer.enhance_resume_for_template(
                    resume_content=resume_text,
                    job_description=job_description,
                    preferences=preferences,
                )
                logger.debug("Successfully enhanced resume content")

                # Validate enhancement result
                if (
                    not enhancement_result
                    or "enhanced_resume_docx" not in enhancement_result
                ):
                    raise ValueError("Enhancement result is missing required data")

                # Get template data
                template_data = enhancement_result["enhanced_resume_docx"]

            except Exception as e:
                logger.error(f"Enhancement process failed: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Resume enhancement failed: {str(e)}",
                )

            # Step 5: Generate DOCX using template handler
            try:
                logger.debug("Starting DOCX generation")
                response = await template_handler.generate_resume_docx(
                    resume_data=template_data, background_tasks=background_tasks
                )
                logger.info("Successfully generated enhanced resume DOCX")
                return response

            except ValueError as ve:
                # Handle validation errors from template handler
                logger.error(f"Template validation error: {str(ve)}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve)
                )
            except Exception as e:
                logger.error(f"DOCX generation failed: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to generate DOCX: {str(e)}",
                )

        except ValueError as ve:
            logger.error(f"File processing error: {str(ve)}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error("Unexpected error in resume generation", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )
