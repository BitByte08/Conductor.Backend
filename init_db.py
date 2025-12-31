"""
Small helper to initialize the database tables from models.
Usage: python init_db.py
"""
from database import engine
import models

print("Creating database tables (if not exists)...")
models.Base.metadata.create_all(bind=engine)
print("Done.")
