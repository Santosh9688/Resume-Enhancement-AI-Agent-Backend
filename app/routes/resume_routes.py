from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from typing import Optional
from app.database import get_db
from app.models.models import Resume, User  # Add User to the import statement
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/")
async def get_resumes(
    user_id: Optional[int] = Query(None, ge=1, description="Filter resumes by user ID"),
    min_ats_score: Optional[float] = Query(None, ge=0, le=100, description="Minimum ATS score"),
    version: Optional[int] = Query(None, ge=1, description="Filter by resume version"),
    db: Session = Depends(get_db)
):
    """
    Retrieves resumes with comprehensive filtering options.
    """
    try:
        # Build base query with joins for efficient loading
        query = db.query(Resume).join(User)
        
        # Apply filters
        if user_id:
            query = query.filter(Resume.UserID == user_id)
        if min_ats_score:
            query = query.filter(Resume.ATSScore >= min_ats_score)  
        if version:
            query = query.filter(Resume.Version == version)

        resumes = query.all()
        
        if not resumes:
            return {
                "status": "success",
                "message": "No resumes found matching the criteria",
                "resumes": []
            }
        
        return {
            "status": "success",
            "total_resumes": len(resumes),
            "resumes": [{
                "id": resume.ResumeID,
                "user": {
                    "id": resume.user.UserID,
                    "username": resume.user.Username
                },
                "version": resume.Version,
                "ats_score": float(resume.ATSScore) if resume.ATSScore else None,
                "content": resume.ResumeContent,
                "content_preview": resume.ResumeContent[:200] + "..." if resume.ResumeContent else None,
                "created_date": resume.CreatedDate.isoformat(),
                "last_modified": resume.LastModifiedDate.isoformat()
            } for resume in resumes]
        }
    except SQLAlchemyError as e:
        logger.error(f"Database error in get_resumes: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error while retrieving resumes"  
        )