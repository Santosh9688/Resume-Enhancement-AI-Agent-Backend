import pyodbc
import logging
from app.database import verify_connection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_direct_connection():
    """
    Test direct connection to SQL Server without SQLAlchemy
    """
    try:
        # List available drivers
        drivers = pyodbc.drivers()
        logger.info("Available ODBC Drivers:")
        for driver in drivers:
            logger.info(f"  - {driver}")

        # Connection string
        conn_str = (
            "DRIVER={SQL Server};"
            "SERVER=LAPTOP-6ST82K8K;"
            "DATABASE=Resume_ai_Agent;"
            "Trusted_Connection=yes;"
        )
        
        # Try to connect
        logger.info("Testing direct connection...")
        conn = pyodbc.connect(conn_str)
        
        # Test the connection
        cursor = conn.cursor()
        cursor.execute("SELECT @@VERSION")
        version = cursor.fetchone()[0]
        
        logger.info("✅ Direct connection successful!")
        logger.info(f"SQL Server Version: {version}")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Connection test failed: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting connection tests...")
    
    # Test direct connection first
    direct_result = test_direct_connection()
    logger.info("Direct connection test result:", "Success" if direct_result else "Failed")
    
    # Test SQLAlchemy connection
    sqlalchemy_result = verify_connection()
    logger.info("SQLAlchemy connection test result:", "Success" if sqlalchemy_result else "Failed")