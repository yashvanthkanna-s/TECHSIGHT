"""
config.py — Central configuration for TechSight.

All tunable parameters live here: search terms, year ranges,
API settings, Trend Score weights, and forecasting parameters.
Import this wherever you need a setting instead of hardcoding values.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# API credentials (loaded from .env, never hardcoded)
# ---------------------------------------------------------------------------
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "")
OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL", "")  # Polite pool identifier
USPTO_API_KEY = os.getenv("USPTO_API_KEY", "")
EPO_CONSUMER_KEY = os.getenv("EPO_CONSUMER_KEY", "")
EPO_CONSUMER_SECRET = os.getenv("EPO_CONSUMER_SECRET", "")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "techsight")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

# ---------------------------------------------------------------------------
# Search domains and terms
# Each domain has a list of search phrases used for both OpenAlex and patents.
# ---------------------------------------------------------------------------
SEARCH_DOMAINS = {
    "solid_state_batteries": [
        "solid state battery",
        "solid state electrolyte",
        "all solid state battery",
        "solid electrolyte interface",
    ],
    "quantum_computing": [
        "quantum computing",
        "quantum computer",
        "quantum algorithm",
        "quantum error correction",
        "quantum gate",
    ],
    "generative_ai": [
        "generative artificial intelligence",
        "generative AI",
        "large language model",
        "text generation model",
        "diffusion model",
    ],
}

# ---------------------------------------------------------------------------
# Year range for data collection
# END_YEAR should be the last *complete* year with reliable data.
# Patent data often has an 18+ month indexing delay — we set a conservative
# cutoff and flag incomplete recent years in the dashboard.
# ---------------------------------------------------------------------------
START_YEAR = 2012
END_YEAR = 2026  # Extended to 2026 to include live current-year research records

# Years we consider potentially incomplete due to patent indexing delays (18-month publication rule)
INCOMPLETE_YEARS_WARNING = [2025, 2026]

# ---------------------------------------------------------------------------
# Data volume caps (per domain per source)
# Start small; expand only after the first end-to-end run works.
# ---------------------------------------------------------------------------
MAX_RECORDS_PER_DOMAIN = 5000

# ---------------------------------------------------------------------------
# API request settings
# ---------------------------------------------------------------------------
REQUEST_TIMEOUT = 30          # seconds
MAX_RETRIES = 3               # retry temporary failures
RETRY_DELAY = 2               # seconds between retries
RATE_LIMIT_DELAY = 0.2        # seconds between API calls (politeness)
OPENALEX_PER_PAGE = 100       # max allowed by OpenAlex

# ---------------------------------------------------------------------------
# NLP settings
# ---------------------------------------------------------------------------
TFIDF_MAX_FEATURES = 5000     # vocabulary size for TF-IDF
TFIDF_NGRAM_RANGE = (1, 2)    # unigrams and bigrams
TFIDF_MIN_DF = 3              # ignore very rare terms
TFIDF_MAX_DF = 0.85           # ignore terms in >85% of documents

# KMeans cluster counts to try during topic exploration
KMEANS_CLUSTER_RANGE = [5, 8, 10, 12, 15]
KMEANS_RANDOM_STATE = 42      # reproducibility

# ---------------------------------------------------------------------------
# Trend Score weights (must sum to 1.0)
# See docs/trend_score.md for component definitions.
# ---------------------------------------------------------------------------
TREND_SCORE_WEIGHTS = {
    "growth_rate": 0.35,
    "consistency": 0.25,
    "acceleration": 0.20,
    "activity_volume": 0.10,
    "citation_signal": 0.10,
}

# Number of recent complete years used for Trend Score calculation
TREND_SCORE_LOOKBACK_YEARS = 5

# Minimum total records (patents + papers) over the lookback period
# for a topic to receive a Trend Score. Prevents tiny topics from
# being labelled "emerging" just because they went from 1 to 3 records.
TREND_SCORE_MIN_ACTIVITY = 10

# ---------------------------------------------------------------------------
# Trend labels and thresholds
# Applied after the 0-100 scaled Trend Score is computed.
# ---------------------------------------------------------------------------
TREND_LABELS = {
    "emerging":       (75, 100),   # Strong consistent growth
    "growing":        (55, 75),    # Moderate positive trend
    "stable":         (40, 55),    # Flat or very slow change
    "declining":      (0, 40),     # Negative trend
}
# "unusual_spike" is assigned separately when a single year shows
# >3x the median activity but surrounding years do not confirm growth.

# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------
FORECAST_HORIZON = 3           # predict this many years ahead
MOVING_AVG_WINDOW = 3          # window size for moving average baseline
TEST_YEARS = 3                 # hold out last N years for evaluation

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
DATA_RAW_DIR = os.path.join("data", "raw")
DATA_PROCESSED_DIR = os.path.join("data", "processed")
MOCK_DIR = "mock"
FALLBACK_DATA_DIR = os.path.join("data", "fallback")

# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))

# CORS origins allowed to call the API (Streamlit runs on 8501 by default)
CORS_ORIGINS = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
]
