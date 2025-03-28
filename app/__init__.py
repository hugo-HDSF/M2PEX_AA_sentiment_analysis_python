from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import logging
import time
import os
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Log database connection string (remove sensitive info)
db_host = os.environ.get('MYSQL_HOST', 'not set')
db_user = os.environ.get('MYSQL_USER', 'not set')
db_name = os.environ.get('MYSQL_DB', 'not set')
logger.info(f"Attempting to connect to database: {db_user}@{db_host}/{db_name}")

app.config.from_object(Config)

# Enable CORS
CORS(app)

# Initialize database
db = SQLAlchemy(app)

# Import routes after initializing app to avoid circular imports
from app.api import routes
from app.database.Tweet import model


# Function to initialize database
def init_db():
    max_attempts = 10
    attempt = 0

    while attempt < max_attempts:
        try:
            logger.info(f"Creating database tables (attempt {attempt + 1}/{max_attempts})")
            db.create_all()
            logger.info("Database tables created successfully")
            return True
        except Exception as e:
            logger.error(f"Error creating database tables: {str(e)}")
            attempt += 1
            time.sleep(2)

    logger.error("Failed to create database tables after multiple attempts")
    return False


# Initialize database tables
with app.app_context():
    init_db()
    logger.info("Database initialized")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
