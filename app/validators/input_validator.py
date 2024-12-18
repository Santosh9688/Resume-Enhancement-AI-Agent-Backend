"""
Handles validation of all input parameters for the resume enhancement system.
This includes job descriptions, user IDs, and optional fields like job titles and keywords.
"""

from pydantic import BaseModel, Field, field_validator, validator
from typing import Optional
import re


class EnhancementRequest(BaseModel):
    """
    Validates the complete enhancement request with all its fields.
    This is our primary input validation model.
    """

    job_description: str = Field(
        ...,
        min_length=50,
        max_length=50000,
        description="Job description with required sections",
    )
    user_id: int = Field(..., gt=0)
    preferred_job_titles: Optional[str] = Field(None, max_length=500)
    target_industries: Optional[str] = Field(None, max_length=500)
    keywords: Optional[str] = Field(None, max_length=1000)

    @field_validator("job_description")
    def validate_job_description(cls, v):
        # Job description validation logic here
        pass

    @field_validator("preferred_job_titles", "target_industries", "keywords")
    def validate_list_fields(cls, v):
        # List fields validation logic here
        pass
