import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===== Original single-ticker config used by script functions =====
TICKER = "SPY"
EXPIRATION_WEIGHTS = [0.5, 0.3, 0.2]
MAX_DISTANCE = 12
NUM_LEVELS = 3
REFRESH_SECONDS = 300  # 5 minutes

# ===== Dashboard defaults =====
DEFAULT_TICKERS = ["SPY", "QQQ"]
DEFAULT_EXPIRATION_WEIGHTS = EXPIRATION_WEIGHTS
DEFAULT_MAX_DISTANCE = MAX_DISTANCE
DEFAULT_NUM_LEVELS = NUM_LEVELS

NY_TIMEZONE = "America/New_York"

DATA_CACHE_DIR = os.path.join(BASE_DIR, "data_cache")
SETTINGS_FILE = os.path.join(DATA_CACHE_DIR, "app_settings.json")
REFRESH_STATUS_FILE = os.path.join(DATA_CACHE_DIR, "refresh_status.json")