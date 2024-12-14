from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine import URL
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('database.log')
    ]
)
logger = logging.getLogger(__name__)

# Database configuration
SERVER = "LAPTOP-6ST82K8K"
DATABASE = "Resume_ai_Agent"
CONNECT_TIMEOUT = 30
QUERY_TIMEOUT = 60

def create_connection_url():
    """Creates a connection URL with proper encoding and timeouts."""
    connection_url = URL.create(
        "mssql+pyodbc",
        query={
            "driver": "SQL Server",
            "server": SERVER,
            "database": DATABASE,
            "Trusted_Connection": "yes",
            "TrustServerCertificate": "yes",
            "timeout": str(CONNECT_TIMEOUT)
        }
    )
    return connection_url

# Create engine with optimized settings
engine = create_engine(
    create_connection_url(),
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=3600,
    echo=True
)

# Session management
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False
)

Base = declarative_base()

def get_db():
    """Database session dependency with enhanced error handling."""
    db = SessionLocal()
    try:
        # Verify connection is active
        db.execute(text("SELECT 1"))
        yield db
    except Exception as e:
        logger.error(f"Database session error: {str(e)}")
        raise
    finally:
        db.close()

def verify_connection():
    """Verifies database connection with detailed diagnostics."""
    try:
        with engine.connect() as connection:
            # Test basic connectivity
            version = connection.execute(text("SELECT @@VERSION")).scalar()
            # Get database details
            result = connection.execute(text("""
                SELECT 
                    DB_NAME() as database_name,
                    CURRENT_USER as username,
                    SERVERPROPERTY('ProductVersion') as version,
                    SERVERPROPERTY('Edition') as edition
            """)).first()
            
            logger.info(f"Connected to database: {result.database_name}")
            logger.info(f"Using SQL Server {result.version} {result.edition}")
            logger.info(f"Authenticated as: {result.username}")
            return True
    except Exception as e:
        logger.error(f"Connection verification failed: {str(e)}")
        return False