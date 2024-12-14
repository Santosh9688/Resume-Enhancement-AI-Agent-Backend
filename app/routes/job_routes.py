from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from typing import Optional
from app.database import get_db
from app.models.models import JobDescription
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/")
async def get_job_descriptions(
    company: Optional[str] = Query(None, description="Filter by company name"),
    title: Optional[str] = Query(None, description="Filter by job title"), 
    keyword: Optional[str] = Query(None, description="Filter by keyword"),
    db: Session = Depends(get_db)
):
    """
    Retrieves job descriptions with multiple filtering options.
    """
    try:
        query = db.query(JobDescription)
        
        # Apply filters
        if company:
            query = query.filter(JobDescription.CompanyName.ilike(f"%{company}%"))
        if title:
            query = query.filter(JobDescription.JobTitle.ilike(f"%{title}%"))
        if keyword:  
            query = query.filter(JobDescription.Keywords.ilike(f"%{keyword}%"))

        jobs = query.all()
        
        return {
            "status": "success",
            "total_jobs": len(jobs),
            "jobs": [{
                "id": job.JobDescriptionID,
                "title": job.JobTitle,
                "company": job.CompanyName,
                "description": job.Description,
                "description_preview": job.Description[:200] + "..." if job.Description else None, 
                "keywords": job.Keywords.split(",") if job.Keywords else [],
                "created_date": job.CreatedDate.isoformat()
            } for job in jobs]
        }
    except SQLAlchemyError as e:
        logger.error(f"Database error in get_job_descriptions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error while retrieving job descriptions"
        )