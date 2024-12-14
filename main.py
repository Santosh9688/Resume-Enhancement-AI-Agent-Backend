from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.database import get_db, verify_connection
from app.routes import router as resume_router  # Import our routes
import datetime
import logging
from typing import Dict, Any

# Configure comprehensive logging with both file and console handlers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI with detailed API documentation
app = FastAPI(
    title="Resume Enhancement AI Agent",
    description="""
    AI-powered resume optimization system providing the following features:
    
    * User Management - Handle user accounts and profiles
    * Resume Processing - Store and version control resumes
    * Job Description Analysis - Extract key requirements and skills
    * Resume Enhancement - AI-powered optimization and tracking
    * ATS Compatibility - Ensure resume passes ATS systems
    
    Use the interactive API documentation below to explore and test endpoints.
    """,
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include our router with API version prefix
app.include_router(
    resume_router,
    prefix="/api/v1",
    tags=["Resume Operations"]
)

@app.on_event("startup")
async def startup_event():
    """
    Application startup handler that verifies critical services.
    Ensures database connectivity before accepting requests.
    """
    logger.info("Starting Resume Enhancement AI Backend...")
    try:
        if verify_connection():
            logger.info("✅ Database connection verified successfully!")
        else:
            logger.error("❌ Database connection failed!")
            # You might want to raise an exception here to prevent startup
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise

@app.get("/",
    tags=["Health Check"],
    summary="API Health Check",
    response_description="Basic API status information",
    responses={
        200: {
            "description": "API is running normally",
            "content": {
                "application/json": {
                    "example": {
                        "status": "active",
                        "message": "Resume AI Backend is running!",
                        "version": "1.0.0",
                        "timestamp": "2024-12-13T00:00:00Z"
                    }
                }
            }
        }
    }
)
def read_root() -> Dict[str, Any]:
    """
    Root endpoint providing basic API health check.
    
    Returns:
        Dict containing API status, version, and timestamp
    """
    return {
        "status": "active",
        "message": "Resume AI Backend is running!",
        "version": "1.0.0",
        "timestamp": datetime.datetime.utcnow().isoformat()
    }

@app.get("/db-test",
    tags=["Health Check"],
    summary="Database Connection Test",
    response_description="Detailed database connection information",
    responses={
        200: {
            "description": "Database connection successful",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "database_connected": True,
                        "connection_info": {
                            "database_name": "Resume_ai_Agent",
                            "username": "dbo",
                            "server_name": "LAPTOP-6ST82K8K",
                            "edition": "Developer Edition",
                            "product_version": "16.0.1135.2"
                        }
                    }
                }
            }
        },
        500: {
            "description": "Database connection failed",
            "content": {
                "application/json": {
                    "example": {
                        "status": "error",
                        "detail": "Database connection error"
                    }
                }
            }
        }
    }
)
def test_db(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Test endpoint verifying database connectivity and configuration.
    
    Args:
        db: Database session injected by FastAPI
        
    Returns:
        Dict containing database connection details and status
        
    Raises:
        HTTPException: If database connection fails
    """
    try:
        # Execute diagnostic queries
        version_query = text("SELECT @@VERSION")
        db_info_query = text("""
            SELECT 
                DB_NAME() as database_name,
                CURRENT_USER as username,
                SERVERPROPERTY('ServerName') as server_name,
                SERVERPROPERTY('Edition') as edition,
                SERVERPROPERTY('ProductVersion') as version
        """)

        # Get database information
        version = db.execute(version_query).scalar()
        db_info = db.execute(db_info_query).first()

        return {
            "status": "success",
            "database_connected": True,
            "connection_info": {
                "database_name": db_info.database_name,
                "username": db_info.username,
                "server_name": db_info.server_name,
                "edition": db_info.edition,
                "product_version": db_info.version
            },
            "sql_server_version": version,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Database test failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database connection error: {str(e)}"
        )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc) -> JSONResponse:
    """
    Global exception handler providing consistent error responses.
    Logs all unhandled exceptions for monitoring and debugging.
    
    Args:
        request: The request that caused the exception
        exc: The unhandled exception
        
    Returns:
        JSONResponse with error details and timestamp
    """
    logger.error(f"Global error handler caught: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "message": "An unexpected error occurred",
            "detail": str(exc),
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
    )