import os
import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Get the database URL from environment variables
DATABASE_URL = os.getenv("DATABASE_URL")

# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Create a base class for declarative models
Base = declarative_base()

# Define the EmotionRecord model
class EmotionRecord(Base):
    __tablename__ = "emotion_records"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    image_name = Column(String(255))
    face_count = Column(Integer)
    emotions_detected = Column(Text)  # JSON string of emotions
    confidence_levels = Column(Text)  # JSON string of confidence levels
    notes = Column(Text, nullable=True)
    
    def __repr__(self):
        return f"<EmotionRecord(id={self.id}, timestamp={self.timestamp}, faces={self.face_count})>"

# Create all tables if they don't exist
Base.metadata.create_all(engine)

# Create a session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Function to get a database session
def get_db_session():
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()

# Function to add an emotion record to the database
def add_emotion_record(image_name, face_count, emotions_detected, confidence_levels, notes=None):
    db = get_db_session()
    
    # Create a new record
    new_record = EmotionRecord(
        image_name=image_name,
        face_count=face_count,
        emotions_detected=emotions_detected,
        confidence_levels=confidence_levels,
        notes=notes
    )
    
    # Add and commit the record
    db.add(new_record)
    db.commit()
    
    # Return the record ID
    return new_record.id

# Function to get all emotion records
def get_all_emotion_records():
    db = get_db_session()
    records = db.query(EmotionRecord).order_by(EmotionRecord.timestamp.desc()).all()
    return records

# Function to get a specific emotion record by ID
def get_emotion_record(record_id):
    db = get_db_session()
    record = db.query(EmotionRecord).filter(EmotionRecord.id == record_id).first()
    return record

# Function to delete an emotion record
def delete_emotion_record(record_id):
    db = get_db_session()
    record = db.query(EmotionRecord).filter(EmotionRecord.id == record_id).first()
    
    if record:
        db.delete(record)
        db.commit()
        return True
    
    return False