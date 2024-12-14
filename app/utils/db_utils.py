from sqlalchemy import text
import logging

logger = logging.getLogger(__name__)

def check_database_health():
    """
    Comprehensive database health check function.
    Returns detailed status information about the database.
    """
    from app.database import engine
    
    try:
        with engine.connect() as connection:
            # Check basic connectivity
            version_info = connection.execute(text("""
                SELECT 
                    @@VERSION as version,
                    DB_NAME() as database_name,
                    CURRENT_USER as username,
                    SERVERPROPERTY('Edition') as edition
            """)).first()
            
            # Check database size and space
            db_size = connection.execute(text("""
                SELECT 
                    SUM(size * 8.0 / 1024) as size_mb
                FROM sys.master_files 
                WHERE database_id = DB_ID()
            """)).scalar()
            
            return {
                "status": "healthy",
                "database_name": version_info.database_name,
                "sql_server_version": version_info.version,
                "edition": version_info.edition,
                "current_user": version_info.username,
                "database_size_mb": round(db_size, 2)
            }
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }