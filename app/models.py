from sqlalchemy import Column, Integer, String, DateTime, LargeBinary
from datetime import datetime
from .database import Base

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user = Column(String, index=True)
    message = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    style = Column(String, nullable=True)
    emotion = Column(String, nullable=True)
    emotional_intensity = Column(String, nullable=True)
    topic = Column(String, nullable=True)
    embedding = Column(LargeBinary, nullable=True)
