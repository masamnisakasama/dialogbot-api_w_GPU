# models.py
from sqlalchemy import Column, Integer, String, DateTime, LargeBinary, Text, JSON, func
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


# MLOps用のクラス　特に、マニュアルと自動を切り替えたり、severityを設定したり
# created/started/finished_atもプロパティとして追加
class MLOpsJob(Base):
    __tablename__ = "mlops_job"
    id = Column(Integer, primary_key=True, index=True)
    status = Column(String(16), default="scheduled")      # sxheduled → running → success or failedの過程
    triggered_by = Column(String(16), default="manual")   # manual/auto
    reason = Column(String(128), nullable=True)           # ex. "drift:high"
    severity = Column(String(16), nullable=True)          # low/moderate/high
    metrics = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
