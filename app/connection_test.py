import pyodbc
import logging

# Set up logging to show detailed information
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_sql_connection():
    """
    Test direct connection to SQL Server with detailed error reporting
    """
    try:
        # First, check available drivers
        drivers = pyodbc.drivers()
        logger.info("Available ODBC Drivers:")
        for driver in drivers:
            logger.info(f"  - {driver}")

        # Attempt connection with detailed error handling
        logger.info("Attempting to connect to SQL Server...")
        
        # Connection string using your specific server details
        conn_str = (
            "DRIVER={SQL Server};"
            "SERVER=LAPTOP-6ST82K8K;"
            "DATABASE=Resume_ai_Agent;"
            "Trusted_Connection=yes;"
        )
        
        # Try to establish connection
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        # Test basic query execution
        logger.info("Testing SQL query execution...")
        cursor.execute("SELECT @@VERSION")
        version = cursor.fetchone()[0]
        logger.info(f"SQL Server Version: {version}")
        
        # Get database details
        cursor.execute("""
            SELECT 
                DB_NAME() as DatabaseName,
                SYSTEM_USER as SystemUser,
                @@SERVERNAME as ServerName
        """)
        db_info = cursor.fetchone()
        logger.info(f"Connected to database: {db_info[0]}")
        logger.info(f"Using server: {db_info[2]}")
        logger.info(f"Connected as user: {db_info[1]}")
        
        cursor.close()
        conn.close()
        logger.info("Connection test completed successfully!")
        return True
        
    except pyodbc.Error as e:
        logger.error("SQL Server Connection Error:")
        logger.error(f"Error message: {str(e)}")
        logger.error("Please verify:")
        logger.error("1. SQL Server is running")
        logger.error("2. Server name is correct")
        logger.error("3. Windows Authentication is enabled")
        logger.error("4. You have proper permissions")
        return False
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting SQL Server connection test...")
    test_sql_connection()