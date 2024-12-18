"""
Application configuration management with comprehensive settings support.
This module handles loading and validating all configuration settings from environment variables.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional, Set
import logging
from typing import Union

# Configure logging
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Application settings management using Pydantic.
    This class defines all configuration settings with proper typing and validation.
    """

    # Database settings
    DB_SERVER: str = "LAPTOP-6ST82K8K"
    DB_NAME: str = "Resume_ai_Agent"
    DB_USER: Optional[str] = None
    DB_PASSWORD: Optional[str] = None
    DB_TRUSTED_CONNECTION: str = "yes"  # Added for Windows Authentication

    # API settings
    API_VERSION: str = "v1"
    DEBUG: bool = False

    # AI Service API Keys
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None

    # File upload settings
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB in bytes
    ALLOWED_FILE_TYPES: Set[str] = {"pdf", "docx"}

    # Enhancement settings
    MAX_RETRIES: int = 3
    TIMEOUT_SECONDS: int = 30

    # Security settings
    SECRET_KEY: Optional[str] = None
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    class Config:
        """
        Pydantic settings configuration.
        Defines how environment variables should be processed.
        """

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

        # Allow arbitrary types to support complex configuration values
        arbitrary_types_allowed = True

        # Custom validation
        validate_assignment = True


@lru_cache()
def get_settings() -> Settings:
    """
    Creates and caches application settings.
    Uses LRU cache to avoid reading environment variables multiple times.
    Includes proper error handling and logging.

    Returns:
        Settings: Application configuration settings

    Raises:
        Exception: If settings cannot be loaded properly
    """
    try:
        logger.info("Loading application settings")
        settings = Settings()

        # Log non-sensitive configuration values
        logger.info("Configuration loaded with the following settings:")
        logger.info(f"Database Server: {settings.DB_SERVER}")
        logger.info(f"Database Name: {settings.DB_NAME}")
        logger.info(f"API Version: {settings.API_VERSION}")
        logger.info(f"Debug Mode: {settings.DEBUG}")
        logger.info(
            f"Maximum Upload Size: {settings.MAX_UPLOAD_SIZE // (1024 * 1024)}MB"
        )
        logger.info(f"Allowed File Types: {', '.join(settings.ALLOWED_FILE_TYPES)}")

        # Verify required settings
        if settings.ANTHROPIC_API_KEY is None:
            logger.warning("ANTHROPIC_API_KEY is not set")
        if settings.OPENAI_API_KEY is None:
            logger.warning("OPENAI_API_KEY is not set")

        return settings

    except Exception as e:
        logger.error(f"Failed to load settings: {str(e)}")
        raise


def get_database_settings() -> dict:
    """
    Returns database-specific settings in a format suitable for SQLAlchemy.
    This helper function makes database configuration more manageable.

    Returns:
        dict: Database configuration settings
    """
    settings = get_settings()
    return {
        "server": settings.DB_SERVER,
        "database": settings.DB_NAME,
        "trusted_connection": settings.DB_TRUSTED_CONNECTION,
        "user": settings.DB_USER,
        "password": settings.DB_PASSWORD,
    }
