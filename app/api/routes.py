from flask import jsonify, request, render_template, redirect, url_for, flash
from app import app, logger, db
import json
from app.model.sentiment_model import predict_sentiment, train_model
from app.database.Tweet.model import Tweet
from app.database.Tweet.repository import TweetRepository
from app.scripts.init_db import init_db as initialize_database
from datetime import datetime
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import joblib


@app.route('/debug')
def debug():
    return jsonify({
        "status": "ok",
        "message": "Flask server is running correctly",
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })


@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/dashboard')
def dashboard():
    try:
        # Get model performance metrics if available
        model_metrics = {}

        if os.path.exists('app/static/metrics.json'):
            with open('app/static/metrics.json', 'r') as f:
                model_metrics = json.load(f)

        # Return the dashboard template with metrics
        return render_template('dashboard.html', metrics=model_metrics)
    except Exception as e:
        logger.error(f"Error in dashboard route: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/history')
def history():
    try:
        # Get recent tweets from database
        recent_tweets = TweetRepository.get_100_latest()
        return render_template('history.html', tweets=recent_tweets)
    except Exception as e:
        logger.error(f"Error in history route: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/init', methods=['POST'])
def initialize_system():
    """Initialize the database and train the model"""
    try:
        # Initialize database with sample data
        with app.app_context():
            # Check if we already have data
            existing_tweets = TweetRepository.count()
            if existing_tweets <= 0:
                # Initialize database with sample tweets
                initialize_database()
                logger.info(f"Database initialized with sample tweets")
            else:
                logger.info(f"Database already contains {existing_tweets} tweets, skipping initialization")

            # Train the model with the sample data
            success = train_model()

            if success:
                return jsonify({
                    "status": "success",
                    "message": "Database initialized and model trained successfully!"
                })
            else:
                return jsonify({
                    "status": "partial_success",
                    "message": "Database initialized but model training failed"
                }), 500
    except Exception as e:
        logger.error(f"Error initializing system: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/retrain', methods=['POST'])
def retrain_model():
    """Endpoint to retrain the sentiment analysis model"""
    try:
        logger.info("Retraining model upon user request")
        tweet_count = TweetRepository.count()

        if tweet_count < 10:
            logger.warning(f"Not enough data to train model: {tweet_count} tweets available")
            return jsonify({
                "status": "error",
                "error": f"Not enough data to train model. Need at least 10 tweets, but only {tweet_count} are available."
            }), 400

        # Train the model
        success = train_model()

        if success:
            logger.info("Model retraining completed successfully")
            return jsonify({
                "status": "success",
                "message": "Model successfully retrained with the latest data!"
            })
        else:
            logger.error("Model retraining failed")
            return jsonify({
                "status": "error",
                "error": "Failed to retrain model. Check server logs for details."
            }), 500

    except Exception as e:
        logger.error(f"Error in model retraining endpoint: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route('/api/analyze', methods=['POST'])
def analyze_sentiment():
    try:
        data = request.get_json()

        if not data or 'tweets' not in data:
            return jsonify({'error': 'No tweets provided'}), 400

        tweets = data['tweets']

        if not isinstance(tweets, list):
            return jsonify({'error': 'Tweets must be provided as a list'}), 400

        if len(tweets) == 0:
            return jsonify({'error': 'Empty tweet list provided'}), 400

        # Analyze sentiment for each tweet
        results = {}
        for tweet in tweets:
            if not isinstance(tweet, str):
                continue  # Skip non-string tweets

            score = predict_sentiment(tweet)
            logger.info(f"Analyzing tweet: {tweet} | Sentiment score: {score}")
            results[tweet] = score

            # Store tweet in database for future model training
            positive = 1 if score > 0 else 0
            negative = 1 if score < 0 else 0

            TweetRepository.add(
                Tweet(text=tweet, positive=positive, negative=negative)
            )

        db.session.commit()
        logger.info(f"Analyzed {len(results)} tweets")

        return jsonify(results)

    except Exception as e:
        logger.error(f"Error in sentiment analysis endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/logs')
def view_logs():
    try:
        with open('app.log', 'r') as f:
            logs = f.readlines()
        return render_template('logs.html', logs=logs)
    except Exception as e:
        logger.error(f"Error in logs route: {str(e)}")
        return jsonify({'error': str(e)}), 500


# Add this route
@app.route('/api/health')
def health_check():
    """API health check endpoint"""
    try:
        return jsonify({
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "db_host": os.environ.get('MYSQL_HOST', 'Not set'),
            "message": "API is up and running"
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500
