import json
from flask import jsonify


def validate_tweet_input(data):
    """
    Validate input for the tweet sentiment analysis endpoint.

    Args:
        data: JSON data from the request

    Returns:
        tuple: (is_valid, error_message or tweets)
    """
    # Check if data is provided
    if not data:
        return False, "No data provided"

    # Check if tweets field exists
    if 'tweets' not in data:
        return False, "Missing 'tweets' field"

    tweets = data['tweets']

    # Check if tweets is a list
    if not isinstance(tweets, list):
        return False, "'tweets' must be a list"

    # Check if tweets list is empty
    if len(tweets) == 0:
        return False, "'tweets' list is empty"

    # Check if all items in the list are strings
    if not all(isinstance(t, str) for t in tweets):
        return False, "All tweets must be strings"

    return True, tweets


def create_error_response(message, status_code):
    """Create a standardized error response."""
    response = jsonify({
        'error': message
    })
    response.status_code = status_code
    return response
