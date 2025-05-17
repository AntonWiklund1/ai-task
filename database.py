import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

# Get the database URL from environment variable
DATABASE_URL = os.getenv("POSTGRESQL_URL")
if not DATABASE_URL:
    raise ValueError("Environment variable 'POSTGRESQL_URL' is not set")

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Create session local class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for declarative models
Base = declarative_base()