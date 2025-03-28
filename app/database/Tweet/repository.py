from app.database.Tweet.model import Tweet
from app import db, logger


class TweetRepository:
    """
    Repository for Tweet model operations with class methods
    """

    @classmethod
    def count(cls):
        """
        Count the number of tweets in the database

        Returns:
            int: Number of tweets
        """
        try:
            count = db.session.query(Tweet).count()
            logger.info(f"Counted {count} tweets in the database")
            return count
        except Exception as e:
            logger.error(f"Error counting tweets: {str(e)}")
            return 0

    @classmethod
    def get_100_latest(cls):
        """
        Get the 100 most recent tweets from the database

        Returns:
            list: List of Tweet objects
        """
        try:
            tweets = Tweet.query.order_by(Tweet.created_at.desc()).limit(100).all()
            logger.info(f"Retrieved {len(tweets)} latest tweets")
            return tweets
        except Exception as e:
            logger.error(f"Error retrieving latest tweets: {str(e)}")
            return []

    @classmethod
    def get_by_id(cls, tweet_id):
        """
        Get a tweet by its ID

        Args:
            tweet_id: ID of the tweet to retrieve

        Returns:
            Tweet: Tweet object if found, None otherwise
        """
        try:
            return Tweet.query.get(tweet_id)
        except Exception as e:
            logger.error(f"Error retrieving tweet with ID {tweet_id}: {str(e)}")
            return None

    @classmethod
    def add(cls, tweet: Tweet):
        """
        Add a new tweet to the database

        Args:
            text (str): Tweet text content
            positive (bool): Positive sentiment flag
            negative (bool): Negative sentiment flag

        Returns:
            Tweet: The created Tweet object, None if an error occurred
        """
        try:
            db.session.add(tweet)
            db.session.commit()
            logger.info(f"Added new tweet: {tweet.text[:30]}...")
            return tweet
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error adding tweet: {str(e)}")
            return None

    @classmethod
    def update(cls, tweet: Tweet, **kwargs):
        """
        Update an existing tweet

        Args:
            tweet_id: ID of the tweet to update
            **kwargs: Tweet attributes to update

        Returns:
            Tweet: The updated Tweet object, None if an error occurred
        """
        try:
            if not tweet:
                logger.warning(f"Tweet with ID {tweet_id} not found for update")
                return None

            for key, value in kwargs.items():
                setattr(tweet, key, value)

            db.session.commit()
            logger.info(f"Updated tweet with ID {tweet_id}")
            return tweet
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error updating tweet with ID {tweet_id}: {str(e)}")
            return None

    @classmethod
    def delete(cls, tweet: Tweet):
        """
        Delete a tweet from the database

        Args:
            tweet_id: ID of the tweet to delete

        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            if not tweet:
                logger.warning(f"Tweet with ID {tweet_id} not found for deletion")
                return False

            db.session.delete(tweet)
            db.session.commit()
            logger.info(f"Deleted tweet with ID {tweet_id}")
            return True
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error deleting tweet with ID {tweet_id}: {str(e)}")
            return False
