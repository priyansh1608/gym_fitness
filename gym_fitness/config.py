from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


DATABASE = BASE_DIR / "database.db"


MODEL_PATH = BASE_DIR / "models" / "trained_model.pkl"


SECRET_KEY = "gym-ai-secret-key"