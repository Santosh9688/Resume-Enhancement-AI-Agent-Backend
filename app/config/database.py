from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os
import logging
import urllib.parse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Database configuration
SERVER = "LAPTOP-6ST82K8K"
DATABASE = "Resume_ai_Agent"

# Create the connection string using ODBC Driver 17
params = urllib.parse.quote_plus(
    'DRIVER={ODBC Driver 17 for SQL Server};'  # Using the newer driver
    f'SERVER={SERVER};'
    f'DATABASE={DATABASE};'
    'Trusted_Connection=yes;'  # Windows Authentication
    'TrustServerCertificate=yes;'  # Trust SSL certificate
)

CONNECTION_STRING = f"mssql+pyodbc:///?odbc_connect={params}"

try:
    # Create the database engine with optimized settings
    engine = create_engine(
        CONNECTION_STRING,
        echo=True,  # SQL query logging (set to False in production)
        pool_pre_ping=True,  # Verify connections before use
        pool_size=5,  # Maintain 5 connections in the pool
        max_overflow=10,  # Allow up to 10 additional connections
        pool_timeout=30  # Connection timeout in seconds
    )
    
    # Configure SQL Server specific optimizations
    @event.listens_for(engine, 'connect')
    def configure_connection(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        # Set session options for better performance
        cursor.execute("SET NOCOUNT ON")
        cursor.execute("SET ARITHABORT ON")
        cursor.close()
    
    # Create session factory
    SessionLocal = sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False
    )
    
    # Create base class for models
    Base = declarative_base()
    
    logger.info("Database configuration initialized successfully")
    
except Exception as e:
    logger.error(f"Error initializing database connection: {str(e)}")
    raise

def get_db():
    """
    Creates a new database session for each request.
    Used as a FastAPI dependency for database access.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_connection():
    """
    Verifies database connection and returns detailed connection information.
    """
    try:
        with engine.connect() as connection:
            # Comprehensive connection test
            result = connection.execute("""
                SELECT 
                    @@VERSION as version,
                    DB_NAME() as database_name,
                    SYSTEM_USER as system_user,
                    SESSION_USER as session_user,
                    @@SPID as session_id
            """).first()
            
            logger.info(f"Connected to database: {result.database_name}")
            logger.info(f"System user: {result.system_user}")
            logger.info(f"Session user: {result.session_user}")
            logger.info(f"Session ID: {result.session_id}")
            
            return True
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        return False