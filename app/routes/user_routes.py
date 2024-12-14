from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from typing import Optional
from app.database import get_db
from app.models.models import User
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get(
    "/",
    summary="Get all users",
    response_description="List of users with pagination",
    responses={
        200: {
            "description": "Successfully retrieved users",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "total_users": 1,
                        "users": [{
                            "id": 1,
                            "username": "test_user",
                            "email": "test@example.com"
                        }]
                    }
                }
            }
        }
    }
)
async def get_users(
    skip: int = Query(0, description="Number of records to skip"),
    limit: int = Query(10, description="Maximum number of records to return"),
    search: Optional[str] = Query(None, description="Search users by username or email"),
    db: Session = Depends(get_db)
):
    """
    Retrieve users with pagination and search capabilities.
    
    Parameters:
    - skip: Number of records to skip for pagination
    - limit: Maximum number of records to return
    - search: Optional search term for username or email
    
    Returns:
    - List of users with their details
    """
    try:
        # Build the base query
        base_query = db.query(User)

        # Apply search filter if provided
        if search:
            base_query = base_query.filter(
                (User.Username.ilike(f"%{search}%")) |
                (User.Email.ilike(f"%{search}%"))
            )

        # Get total count for pagination
        total_users = base_query.count()

        # Create a subquery for pagination
        subquery = base_query.subquery()

        # Apply pagination to the subquery
        users_query = db.query(subquery).order_by(subquery.c.UserID).offset(skip).limit(limit)

        # Execute the query and fetch the users
        users = users_query.all()

        return {
            "status": "success",
            "total_users": total_users,
            "users": [{
                "id": user.UserID,
                "username": user.Username,
                "email": user.Email
            } for user in users]
        }
    except SQLAlchemyError as e:
        logger.error(f"Database error in get_users: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error while retrieving users"
        )