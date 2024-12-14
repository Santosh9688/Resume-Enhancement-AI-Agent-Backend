import pyodbc
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_direct_connection():
    """Test direct connection to SQL Server using pyodbc"""
    try:
        # List available drivers
        logger.info("Available ODBC drivers:")
        for driver in pyodbc.drivers():
            logger.info(f"  - {driver}")

        # Connection string for Windows Authentication
        conn_str = (
            "DRIVER={SQL Server};"
            "SERVER=LAPTOP-6ST82K8K;"
            "DATABASE=Resume_ai_Agent;"
            "Trusted_Connection=yes;"
        )
        
        logger.info("Attempting to connect to database...")
        conn = pyodbc.connect(conn_str)
        
        cursor = conn.cursor()
        cursor.execute("SELECT @@VERSION")
        version = cursor.fetchone()[0]
        
        logger.info("✅ Successfully connected to database!")
        logger.info(f"SQL Server Version: {version}")
        
        # Test database permissions
        cursor.execute("SELECT DB_NAME(), CURRENT_USER, SYSTEM_USER")
        db_info = cursor.fetchone()
        logger.info(f"Database Name: {db_info[0]}")
        logger.info(f"Current User: {db_info[1]}")
        logger.info(f"System User: {db_info[2]}")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_direct_connection()