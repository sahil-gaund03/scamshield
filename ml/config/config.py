import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "datasets"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model & Vectorizer Export Paths
MODEL_DIR = BASE_DIR / "ml" / "models"
VECTORIZER_DIR = BASE_DIR / "ml" / "vectorizers"
LOG_DIR = BASE_DIR / "ml" / "logs"

# Ensure directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VECTORIZER_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Datasets
SMS_SPAM_RAW = RAW_DATA_DIR / "sms_spam.csv"
PHISHING_RAW = RAW_DATA_DIR / "phishing.csv"
CLEANED_SPAM_CSV = PROCESSED_DATA_DIR / "cleaned_spam.csv"

# NLP Configuration
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)

# Logging
LOG_FILE_PATH = LOG_DIR / "scamshield.log"

# Model File Names
TEXT_MODEL_PATH = MODEL_DIR / "scamshield_text_model.pkl"
URL_MODEL_PATH = MODEL_DIR / "scamshield_url_model.pkl"
TEXT_VECTORIZER_PATH = VECTORIZER_DIR / "scamshield_tfidf.pkl"

# Random state for reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
