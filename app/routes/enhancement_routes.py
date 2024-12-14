from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session  
from sqlalchemy.exc import SQLAlchemyError
from typing import Optional
from datetime import datetime  
from app.database import get_db
from app.models.models import ResumeEnhancement, JobDescription
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/")
async def get_enhancements(
    resume_id: Optional[int] = Query(None, ge=1, description="Filter by resume ID"),
    date_after: Optional[datetime] = Query(None, description="Filter by date after"),
    date_before: Optional[datetime] = Query(None, description="Filter by date before"),
    db: Session = Depends(get_db)
):
    """
    Retrieves resume enhancements with comprehensive filtering and error handling.  
    """
    try:
        query = db.query(ResumeEnhancement).join(
            JobDescription,
            ResumeEnhancement.JobDescriptionID == JobDescription.JobDescriptionID,
            isouter=True
        )
        
        # Apply filters
        if resume_id:
            query = query.filter(ResumeEnhancement.ResumeID == resume_id)
        if date_after:
            query = query.filter(ResumeEnhancement.CreatedDate >= date_after)
        if date_before:
            query = query.filter(ResumeEnhancement.CreatedDate <= date_before)

        enhancements = query.all()
        
        if not enhancements:
            return {
                "status": "success", 
                "message": f"No enhancements found{' for resume_id=' + str(resume_id) if resume_id else ''}",
                "enhancements": []
            }
        
        return {
            "status": "success",
            "total_enhancements": len(enhancements),
            "enhancements": [{
                "id": enh.EnhancementID,
                "resume_id": enh.ResumeID,
                "job_description": {
                    "id": enh.JobDescriptionID,
                    "title": enh.job_description.JobTitle if enh.job_description else None,
                    "company": enh.job_description.CompanyName if enh.job_description else None
                },
                "enhanced_content": enh.EnhancedContent,
                "enhanced_content_preview": enh.EnhancedContent[:200] + "..." if enh.EnhancedContent else None,
                "changes_applied": enh.ChangesApplied,
                "created_date": enh.CreatedDate.isoformat()  
            } for enh in enhancements]
        }
    except SQLAlchemyError as e:
        logger.error(f"Database error in get_enhancements: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error while retrieving enhancements"
        )