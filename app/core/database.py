"""
Database configuration and connection management.
This module handles database connections, sessions, and transactions for the application.
"""

from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from urllib.parse import quote_plus
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from typing import Any, Callable, TypeVar, AsyncGenerator, Optional
import pyodbc
import logging

import urllib
from app.core.config import get_settings
from app.core.exceptions import DatabaseError

# Type variable for generic return type
T = TypeVar("T")

# Configure logging
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()


# Create the async database URL
def get_database_url() -> str:
    """
    Constructs the async database URL for SQL Server.

    For SQL Server, we need to use the aioodbc driver for async operations.
    The URL format is different from the sync version to support async operations.
    """
    connection_params = {
        "DRIVER": "{SQL Server}",
        "SERVER": settings.DB_SERVER,
        "DATABASE": settings.DB_NAME,
        "Trusted_Connection": "yes",
        "TrustServerCertificate": "yes",
    }

    # Create the connection string
    conn_str = ";".join(f"{k}={v}" for k, v in connection_params.items())

    # URL encode the connection string
    encoded_conn_str = quote_plus(conn_str)

    # Create the async database URL
    # Note: Using aioodbc for async operations
    DATABASE_URL = f"mssql+aioodbc:///?odbc_connect={encoded_conn_str}"

    logger.info(f"Database URL created for server: {settings.DB_SERVER}")
    return DATABASE_URL


# Create async engine
engine = create_async_engine(
    get_database_url(),
    echo=True,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

# Create async session factory
SessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


class DatabaseConnectionManager:
    """
    Manages database connection configuration and creation.
    This class handles the creation and validation of database connection strings.
    """

    def __init__(self, settings):
        """
        Initialize the connection manager with application settings.

        Args:
            settings: Application configuration settings
        """
        self.settings = settings
        self.connection_params = {
            "DRIVER": "{SQL Server}",  # Using the basic SQL Server driver
            "SERVER": settings.DB_SERVER,
            "DATABASE": settings.DB_NAME,
            "Trusted_Connection": "yes",  # Enables Windows Authentication
            "TrustServerCertificate": "yes",  # Required for secure connection
        }

    def create_connection_string(self) -> str:
        """
        Creates a properly formatted connection string for SQL Server.

        Returns:
            str: The formatted connection string for SQLAlchemy

        Raises:
            DatabaseError: If connection string creation fails
        """
        try:
            # Join the parameters into a connection string
            conn_str = ";".join(f"{k}={v}" for k, v in self.connection_params.items())

            # URL encode for SQLAlchemy
            encoded_conn_str = urllib.parse.quote_plus(conn_str)

            # Log connection details (excluding sensitive information)
            logger.info(
                f"Creating connection string for server: {self.settings.DB_SERVER}, "
                f"database: {self.settings.DB_NAME}"
            )

            # Return the complete connection URL
            return f"mssql+pyodbc:///?odbc_connect={encoded_conn_str}"

        except Exception as e:
            error_msg = f"Failed to create connection string: {str(e)}"
            logger.error(error_msg)
            raise DatabaseError(error_msg)

    async def test_connection(self, engine) -> bool:
        """
        Tests the database connection using the provided engine.

        Args:
            engine: SQLAlchemy engine instance

        Returns:
            bool: True if connection is successful

        Raises:
            DatabaseError: If connection test fails
        """
        try:
            async with engine.connect() as connection:
                # Test basic connectivity
                await connection.execute(text("SELECT 1"))

                # Get database information
                result = await connection.execute(
                    text("""
                    SELECT 
                        DB_NAME() as database_name,
                        CURRENT_USER as username
                    """)
                )
                info = result.first()

                logger.info("Database connection test successful:")
                logger.info(f"Connected to database: {info.database_name}")
                logger.info(f"Connected as user: {info.username}")

                return True

        except Exception as e:
            error_msg = f"Database connection test failed: {str(e)}"
            logger.error(error_msg)
            raise DatabaseError(error_msg)


class TransactionManager:
    """Manages database transactions with proper error handling and retries."""

    def __init__(self, db: Session, max_retries: int = 3):
        self.db = db
        self.max_retries = max_retries
        logger.debug(f"TransactionManager initialized with session id: {id(db)}")

    async def execute_operation(
        self,
        operation: Callable[..., T],
        error_message: str = "Database operation failed",
        *args: Any,
        **kwargs: Any,
    ) -> T:
        retry_count = 0
        last_error = None

        logger.debug(f"Starting execute_operation with session id: {id(self.db)}")
        logger.debug(f"Current transaction status: {self.db.in_transaction()}")

        while retry_count < self.max_retries:
            try:
                # Check transaction state before beginning
                if self.db.in_transaction():
                    logger.warning("Transaction already in progress, rolling back")
                    await self.db.rollback()

                logger.debug("Beginning new transaction")
                async with self.db.begin():
                    logger.debug(f"Executing operation {operation.__name__}")
                    result = await operation(*args, **kwargs)
                    await self.db.commit()
                    logger.debug("Database operation completed successfully")
                    return result

            except SQLAlchemyError as e:
                last_error = e
                retry_count += 1
                logger.warning(
                    f"Database operation failed (attempt {retry_count}): {str(e)}"
                )
                try:
                    if self.db.in_transaction():
                        await self.db.rollback()
                        logger.debug("Transaction rolled back successfully")
                except Exception as rollback_error:
                    logger.error(f"Rollback failed: {str(rollback_error)}")

                if retry_count >= self.max_retries:
                    break

        error_details = f"{error_message}: {str(last_error)}"
        logger.error(error_details)
        raise DatabaseError(error_details)


# Initialize database components
try:
    # Create connection manager and get connection string
    db_manager = DatabaseConnectionManager(settings)
    DATABASE_URL = db_manager.create_connection_string()

    # Initialize the SQLAlchemy engine
    engine = create_engine(
        DATABASE_URL,
        echo=True,  # Show SQL queries for debugging
        pool_pre_ping=True,  # Check connection before using
        pool_size=5,  # Keep 5 connections in pool
        max_overflow=10,  # Allow up to 10 extra connections
    )

    # Test the connection
    db_manager.test_connection(engine)

    # Create session factory
    SessionLocal = sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
        class_=Session,
    )

    # Create base class for SQLAlchemy models
    Base = declarative_base()
    logger.info("Database configuration completed successfully")

except Exception as e:
    logger.error(f"Database initialization failed: {str(e)}")
    logger.error("Please verify SQL Server configuration and permissions")
    raise


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    logger.debug("Starting new database session creation")
    try:
        async with SessionLocal() as session:
            logger.debug(f"Database session created with ID: {id(session)}")
            # Verify session is working
            await session.execute(text("SELECT 1"))
            logger.debug("Session test query executed successfully")
            yield session
            logger.debug(f"Session {id(session)} yielded successfully")
    except Exception as e:
        logger.error(f"Session error occurred: {str(e)}")
        raise
    finally:
        logger.debug("Cleaning up database session")


async def verify_connection() -> bool:
    """
    Performs a comprehensive check of the database connection.
    This can be used for health checks or connection verification.
    """
    try:
        db_manager = DatabaseConnectionManager(settings)
        return await db_manager.test_connection(engine)
    except Exception as e:
        logger.error(f"Connection verification failed: {str(e)}")
        return False
