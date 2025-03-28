#!/usr/bin/env python
import sys
from app import app, logger
from app.database.Tweet.repository import TweetRepository
from app.model.sentiment_model import train_model
from app.scripts.init_db import init_db


def setup_database():
    """Initialize database with sample data"""
    with app.app_context():
        # Check if we already have data
        existing_tweets = TweetRepository.count()

        if existing_tweets > 0:
            logger.info(f"Database already contains {existing_tweets} tweets, skipping initialization")
            return True

        # Initialize database with sample tweets
        count = init_db()
        logger.info(f"Database initialized with {count} sample tweets")
        return True


def train_initial_model():
    """Train the initial model"""
    with app.app_context():
        success = train_model()
        if success:
            logger.info("Initial model training completed successfully")
            return True
        else:
            logger.error("Initial model training failed")
            return False


def retrain_model():
    """Retrain the model with current data"""
    with app.app_context():
        # Verify we have enough data to train
        tweet_count = TweetRepository.count()

        if tweet_count < 10:
            logger.warning(f"Not enough data to retrain model: {tweet_count} tweets available")
            logger.warning("Need at least 10 tweets for retraining")
            return False

        logger.info(f"Retraining model with {tweet_count} tweets...")
        success = train_model()

        if success:
            logger.info("Model retraining completed successfully")
            return True
        else:
            logger.error("Model retraining failed")
            return False


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "all"

    if action == "db" or action == "all":
        setup_database()

    if action == "model" or action == "all":
        train_initial_model()

    if action == "retrain":
        retrain_model()
