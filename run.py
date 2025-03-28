from app import app, logger
import time
import mysql.connector
import os


def wait_for_db():
    """Wait for database to be ready"""
    logger.info("Checking database connection...")
    db_host = os.environ.get('MYSQL_HOST', 'db')
    db_user = os.environ.get('MYSQL_USER', 'user')
    db_pass = os.environ.get('MYSQL_PASSWORD', 'password')
    db_name = os.environ.get('MYSQL_DB', 'sentiment_analysis')

    max_attempts = 30
    attempt = 0

    while attempt < max_attempts:
        try:
            conn = mysql.connector.connect(
                host=db_host,
                user=db_user,
                password=db_pass,
                database=db_name
            )
            conn.close()
            logger.info("Database connection successful!")
            return True
        except mysql.connector.Error as err:
            attempt += 1
            logger.warning(f"Database connection attempt {attempt}/{max_attempts} failed: {err}")
            time.sleep(2)

    logger.error("Could not connect to database after multiple attempts")
    return False


if __name__ == "__main__":
    # Wait for database to be ready
    if wait_for_db():
        logger.info("Starting Flask application on 0.0.0.0:5001")
        app.run(host='0.0.0.0', port=5001, debug=True)
    else:
        logger.error("Exiting due to database connection failure")
