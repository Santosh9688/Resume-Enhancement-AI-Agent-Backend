import logging
from sqlalchemy import text
from app.database import engine

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_database_connection():
    """
    Tests the database connection with clear status reporting.
    """
    logger.info("Starting database connection test...")
    
    try:
        # Test 1: Basic Connection
        with engine.connect() as connection:
            logger.info("Test 1: Basic connectivity")
            version = connection.execute(text("SELECT @@VERSION")).scalar()
            logger.info("✓ Basic connection successful")
            logger.info(f"SQL Server Version: {version}")

            # Test 2: Database Information
            logger.info("\nTest 2: Database information")
            result = connection.execute(text("""
                SELECT DB_NAME() as database_name
            """)).first()
            logger.info(f"✓ Connected to database: {result.database_name}")

            # Test 3: User Information
            logger.info("\nTest 3: User verification")
            result = connection.execute(text("""
                SELECT CURRENT_USER as username
            """)).first()
            logger.info(f"✓ Connected as user: {result.username}")

            logger.info("\n✓ All connection tests passed successfully!")
            return True

    except Exception as e:
        logger.error(f"❌ Database connection test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_database_connection()