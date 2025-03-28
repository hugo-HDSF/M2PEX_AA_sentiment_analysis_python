from app import db
from datetime import datetime


class Tweet(db.Model):
    __tablename__ = 'tweets'

    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    positive = db.Column(db.Integer, nullable=False, default=0)  # 1 if positive, 0 otherwise
    negative = db.Column(db.Integer, nullable=False, default=0)  # 1 if negative, 0 otherwise
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Tweet {self.id}: {self.text[:20]}...>"
