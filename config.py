from datetime import datetime
from pathlib import Path

# Start date for the entire project
PROJECT_START_DATE = datetime(2025, 3, 15)

# Directory for storing artifacts (models, plots, etc.)
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

# S3 object keys for model artifacts
S3_MODEL_KEY = "model.joblib"
S3_STATS_KEY = "stats.json"

