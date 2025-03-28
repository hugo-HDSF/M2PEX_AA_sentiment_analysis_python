import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import joblib
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from app import logger
from app.database.Tweet.repository import TweetRepository
from app.scripts.pdf_generator import generate_evaluation_report
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize

nltk.download('stopwords')
nltk.download('punkt_tab')

# Define paths for model and vectorizer
MODEL_PATH = 'app/data/sentiment_model.pkl'
VECTORIZER_PATH = 'app/data/tfidf_vectorizer.pkl'
METRICS_PATH = 'app/static/metrics.json'
CONFUSION_MATRIX_PATH = 'app/static/confusion_matrix.png'


def remove_stopwords(text, language='french'):
    """
    Remove stopwords from text using NLTK

    Args:
        text (str): Input text
        language (str): Language for stopwords (default: french)

    Returns:
        str: Text with stopwords removed
    """
    # Handle None or non-string values
    if not isinstance(text, str):
        return ""

    stop_words = set(stopwords.words(language))
    word_tokens = word_tokenize(text.lower(), language=language)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)


def clean_string(text):
    """
    Clean the input string by removing special characters and extra spaces.

    Args:
        text (str): The input string to clean.
    Returns:
        str: The cleaned string.
    """
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    # Then, replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    # Optional: strip leading and trailing spaces
    text = text.strip()

    return text


def train_model():
    """
    Train or retrain the sentiment analysis model using data from the database.
    """
    logger.info("Starting model training/retraining")

    try:
        # Fetch data from database
        tweets = TweetRepository.get_100_latest()

        if len(tweets) < 10:
            logger.warning("Not enough data to train model")
            return False

        df = pd.DataFrame([(t.text, t.positive, t.negative) for t in tweets],
                          columns=['text', 'positive', 'negative'])

        # Create a single sentiment target: 1 for positive, 0 for negative
        # Using positive column as the target since we're simplifying to binary classification
        df['sentiment'] = df['positive']

        # Apply stopwords removal - using both French and English stop words
        logger.info("Preprocessing text, cleaning and removing stopwords")
        df['processed_text'] = df['text'].apply(lambda x: remove_stopwords(clean_string(x), 'french'))

        # Prepare data
        X = df['processed_text']
        y = df['sentiment']

        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(max_features=1000)
        X_tfidf = vectorizer.fit_transform(X)

        # Split data 80/20
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y, test_size=0.2, random_state=42)

        # Train a single logistic regression model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)

        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Save confusion matrix as image
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix for Sentiment Classification')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(CONFUSION_MATRIX_PATH)

        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        accuracy = accuracy_score(y_test, y_pred)

        metrics = {
            'sentiment': {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'accuracy': float(accuracy)
            },
            'last_trained': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Create figure for confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Sentiment Analysis Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig('app/static/confusion_matrix.png')
        plt.close()

        # Save evaluation metrics
        with open('app/static/metrics.json', 'w') as f:
            json.dump(metrics, f)

        # Save metrics
        with open(METRICS_PATH, 'w') as f:
            json.dump(metrics, f)

        # Save model and vectorizer
        joblib.dump(model, MODEL_PATH)
        joblib.dump(vectorizer, VECTORIZER_PATH)

        generate_evaluation_report(
            y_test,
            y_pred,
            metrics
        )

        logger.info("Model training completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        return False


def predict_sentiment(text):
    """
    Predict the sentiment of the given text.

    Args:
        text (str): The text to analyze

    Returns:
        float: A sentiment score between -1 (very negative) and 1 (very positive)
    """
    # Check if model exists, if not train it
    if not os.path.exists(MODEL_PATH):
        if not train_model():
            # If training failed, return neutral sentiment
            return 0

    # Load model
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)

        # Preprocess text by removing stopwords (same as in training)
        processed_text = remove_stopwords(text, 'french')
        processed_text = clean_string(processed_text)

        # Vectorize text
        X = vectorizer.transform([processed_text])

        # Predict sentiment
        # Get probability of positive class (index 1)
        probabilities = model.predict_proba(X)[0]

        if len(probabilities) > 1:
            # Convert probability to a score between -1 and 1
            # The probability of positive class minus 0.5, then scaled by 2
            sentiment_score = (probabilities[1] - 0.5) * 2
        else:
            # Binary prediction (if model doesn't output probabilities)
            prediction = model.predict(X)[0]
            sentiment_score = 1.0 if prediction == 1 else -1.0

        logger.info(f"Sentiment analysis: '{text}' - score: {sentiment_score}")

        return sentiment_score
    except Exception as e:
        logger.error(f"Error predicting sentiment: {str(e)}")
        return 0  # Return neutral sentiment if there's an error
