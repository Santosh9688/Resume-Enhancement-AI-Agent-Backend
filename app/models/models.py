from sqlalchemy import (
    Column, Integer, String, DateTime, Float, ForeignKey, 
    DECIMAL, Index, UniqueConstraint, CheckConstraint, Text
)
from sqlalchemy.orm import relationship, validates
from datetime import datetime
from app.database import Base
import re

class User(Base):
    """
    Model representing users in our system.
    Stores basic user information and manages relationships with resumes and job descriptions.
    Includes validation for email format and username requirements.
    """
    __tablename__ = "Users"

    # Primary key with auto-increment
    UserID = Column(Integer, primary_key=True, autoincrement=True)
    
    # Core fields with constraints
    Username = Column(String(100), nullable=False, unique=True)
    Email = Column(String(255), nullable=False, unique=True)
    
    # Timestamps for tracking
    CreatedDate = Column(DateTime, default=datetime.utcnow, nullable=False)
    LastModifiedDate = Column(
        DateTime, 
        default=datetime.utcnow, 
        onupdate=datetime.utcnow,
        nullable=False
    )

    # Relationships with cascading delete
    resumes = relationship(
        "Resume",
        back_populates="user",
        cascade="all, delete-orphan"
    )
    job_descriptions = relationship(
        "JobDescription",
        back_populates="user",
        cascade="all, delete-orphan"
    )

    # Adding indexes for frequently queried fields
    __table_args__ = (
        Index('idx_user_email', Email),
        Index('idx_user_username', Username),
    )

    @validates('Email')
    def validate_email(self, key, email):
        """Validates email format using regex pattern."""
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
            raise ValueError('Invalid email format')
        return email

    @validates('Username')
    def validate_username(self, key, username):
        """Validates username length and characters."""
        if not 3 <= len(username) <= 100:
            raise ValueError('Username must be between 3 and 100 characters')
        if not re.match(r'^[a-zA-Z0-9_-]+$', username):
            raise ValueError('Username can only contain letters, numbers, underscores, and hyphens')
        return username

class Resume(Base):
    """
    Model for storing resume versions.
    Includes version control and ATS scoring functionality.
    Manages the relationship between users and their resume versions.
    """
    __tablename__ = "Resumes"

    ResumeID = Column(Integer, primary_key=True, autoincrement=True)
    UserID = Column(
        Integer, 
        ForeignKey('Users.UserID', ondelete='CASCADE'),
        nullable=False
    )
    
    # Using Text instead of String for unlimited length
    ResumeContent = Column(Text, nullable=False)
    Version = Column(Integer, nullable=False, default=1)
    
    # ATS Score with validation constraint
    ATSScore = Column(
        DECIMAL(5,2),
        CheckConstraint('ATSScore >= 0 AND ATSScore <= 100'),
        nullable=True
    )
    
    CreatedDate = Column(DateTime, default=datetime.utcnow, nullable=False)
    LastModifiedDate = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )

    # Relationships
    user = relationship("User", back_populates="resumes")
    enhancements = relationship(
        "ResumeEnhancement",
        back_populates="resume",
        cascade="all, delete-orphan"
    )

    # Indexes and constraints
    __table_args__ = (
        Index('idx_resume_user_version', UserID, Version),
        UniqueConstraint('UserID', 'Version', name='uq_user_resume_version'),
    )

    @validates('ATSScore')
    def validate_ats_score(self, key, score):
        """Validates ATS score range."""
        if score is not None and not (0 <= float(score) <= 100):
            raise ValueError('ATS Score must be between 0 and 100')
        return score

class JobDescription(Base):
    """
    Model for storing job descriptions.
    Manages job posting details and keyword matching capabilities.
    Includes validation for required fields and keyword formatting.
    """
    __tablename__ = "JobDescriptions"

    JobDescriptionID = Column(Integer, primary_key=True, autoincrement=True)
    UserID = Column(
        Integer, 
        ForeignKey('Users.UserID', ondelete='CASCADE'),
        nullable=False
    )
    
    JobTitle = Column(String(255), nullable=False)
    CompanyName = Column(String(255), nullable=False)
    Description = Column(Text, nullable=False)
    Keywords = Column(Text, nullable=True)  # Stored as comma-separated values
    CreatedDate = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    user = relationship("User", back_populates="job_descriptions")
    enhancements = relationship(
        "ResumeEnhancement",
        back_populates="job_description",
        cascade="all, delete-orphan"
    )

    # Indexes for searching
    __table_args__ = (
        Index('idx_job_company', CompanyName),
        Index('idx_job_title', JobTitle),
    )

    @validates('Keywords')
    def validate_keywords(self, key, keywords):
        """Validates and formats keywords."""
        if keywords:
            # Clean and normalize keywords
            keyword_list = [k.strip().lower() for k in keywords.split(',') if k.strip()]
            return ','.join(sorted(set(keyword_list)))  # Remove duplicates
        return keywords

class ResumeEnhancement(Base):
    """
    Model for tracking resume enhancements.
    Manages the relationship between resumes and job descriptions.
    Tracks changes and improvements made to resumes.
    """
    __tablename__ = "ResumeEnhancements"

    EnhancementID = Column(Integer, primary_key=True, autoincrement=True)
    ResumeID = Column(
        Integer, 
        ForeignKey('Resumes.ResumeID', ondelete='CASCADE'),
        nullable=False
    )
    JobDescriptionID = Column(
        Integer, 
        ForeignKey('JobDescriptions.JobDescriptionID', ondelete='CASCADE'),
        nullable=False
    )
    
    EnhancedContent = Column(Text, nullable=False)
    ChangesApplied = Column(Text, nullable=False)  # Store as JSON string
    CreatedDate = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    resume = relationship("Resume", back_populates="enhancements")
    job_description = relationship("JobDescription", back_populates="enhancements")

    # Indexes for querying
    __table_args__ = (
        Index('idx_enhancement_resume', ResumeID),
        Index('idx_enhancement_job', JobDescriptionID),
        Index('idx_enhancement_date', CreatedDate),
    )