from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()

class FaceEntry(Base):
    __tablename__ = 'faces'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    timestamp = Column(DateTime)

engine = create_engine('sqlite:///faces.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
