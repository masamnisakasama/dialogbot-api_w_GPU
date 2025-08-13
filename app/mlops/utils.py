import os
import pickle
import json
from app import database, models
from sqlalchemy.orm import Session

BASELINE_STATS_PATH = "baseline_stats.json"

def get_db_session():
    return database.SessionLocal()

def load_embeddings_from_db():
    db = get_db_session()
    try:
        conversations = db.query(models.Conversation).filter(models.Conversation.embedding != None).all()
        embeddings = [pickle.loads(conv.embedding) for conv in conversations if conv.embedding]
        return embeddings
    finally:
        db.close()

def save_baseline_stats(stats: dict):
    with open(BASELINE_STATS_PATH, "w") as f:
        json.dump(stats, f)

def load_baseline_stats():
    if not os.path.exists(BASELINE_STATS_PATH):
        return None
    with open(BASELINE_STATS_PATH, "r") as f:
        return json.load(f)
