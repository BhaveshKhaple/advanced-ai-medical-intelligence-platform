from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "predictions.db")
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    image_filename = Column(String, nullable=False)
    gradcam_path = Column(String)
    diagnosis = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    llm_report = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)


def save_prediction(**kwargs):
    db = SessionLocal()
    try:
        rec = Prediction(**kwargs)
        db.add(rec)
        db.commit()
        db.refresh(rec)
        return rec.id
    finally:
        db.close()


def get_history(limit=50):
    db = SessionLocal()
    try:
        return db.query(Prediction).order_by(Prediction.created_at.desc()).limit(limit).all()
    finally:
        db.close()
