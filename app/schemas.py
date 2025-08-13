from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

class ConversationCreate(BaseModel):
    user: str
    message: str

class Conversation(BaseModel):
    id: int
    user: str
    message: str
    timestamp: datetime
    style: Optional[str] = None
    emotion: Optional[str] = None
    emotional_intensity: Optional[str] = None
    topic: Optional[str] = None

    class Config:
        orm_mode = True

class Explanation(BaseModel):
    id: int
    similarity: float
    top_dimensions: List[int]
    explanation_text: Optional[str] = None
