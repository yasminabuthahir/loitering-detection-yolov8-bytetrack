from src.loitering import LoiteringDetector
from src.db import init_db

if __name__ == "__main__":
    init_db()
    detector = LoiteringDetector()
    detector.run()