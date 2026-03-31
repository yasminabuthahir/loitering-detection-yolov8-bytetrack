import sqlite3
import os

DB_PATH = "db/loitering.db"

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER,
            duration REAL,
            timestamp TEXT,
            snapshot_path TEXT
        )
    """)
    conn.commit()
    conn.close()

def insert_event(track_id, duration, timestamp, snapshot_path):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO events (track_id, duration, timestamp, snapshot_path)
        VALUES (?, ?, ?, ?)
    """, (track_id, duration, timestamp, snapshot_path))
    conn.commit()
    conn.close()