import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for the application."""
    # Database configuration
    SQLALCHEMY_DATABASE_URI = f"mysql+mysqlconnector://{os.environ.get('MYSQL_USER')}:{os.environ.get('MYSQL_PASSWORD')}@{os.environ.get('MYSQL_HOST', 'db')}/{os.environ.get('MYSQL_DB')}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Flask configuration
    SECRET_KEY = os.environ.get('FLASK_SECRET_KEY', 'default-secret-key')
    DEBUG = os.environ.get('FLASK_DEBUG', '0') == '1'

    # Application configuration
    MODEL_PATH = 'app/static/sentiment_model.pkl'
    VECTORIZER_PATH = 'app/static/tfidf_vectorizer.pkl'
    METRICS_PATH = 'app/static/metrics.json'
