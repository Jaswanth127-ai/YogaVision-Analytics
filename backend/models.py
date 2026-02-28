from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class YogaSession(Base):
    __tablename__ = "yoga_sessions"

    session_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer)
    pose_name = Column(String(100))
    duration = Column(Float)
    accuracy = Column(Float)
    calories = Column(Float)
    timestamp = Column(DateTime)