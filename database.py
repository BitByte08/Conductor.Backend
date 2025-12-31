import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Use env var if set, otherwise default to a path inside the backend package directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
default_db_path = os.path.join(BASE_DIR, "conductor.db")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{default_db_path}")

# If using a sqlite file path, ensure the parent directory exists and the file is creatable
if DATABASE_URL.startswith("sqlite:///"):
    # extract filesystem path
    fs_path = DATABASE_URL.replace("sqlite:///", "")
    parent = os.path.dirname(fs_path)
    try:
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        # create empty file if it doesn't exist so that permission errors surface early
        if not os.path.exists(fs_path):
            open(fs_path, "a").close()
    except Exception as e:
        raise RuntimeError(f"Unable to prepare SQLite database file at {fs_path}: {e}")

try:
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite:") else {})
except Exception as e:
    raise RuntimeError(f"Failed to create database engine for {DATABASE_URL}: {e}")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
