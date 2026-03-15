# ===== Original single-ticker config used by oi_levels.py, gamma_exposure.py, etc. =====
TICKER = "SPY"
EXPIRATION_WEIGHTS = [0.5, 0.3, 0.2]
MAX_DISTANCE = 12
NUM_LEVELS = 3
REFRESH_SECONDS = 300

# ===== Dashboard/default multi-ticker config =====
DEFAULT_TICKERS = ["SPY", "QQQ"]
DEFAULT_EXPIRATION_WEIGHTS = EXPIRATION_WEIGHTS
DEFAULT_MAX_DISTANCE = MAX_DISTANCE
DEFAULT_NUM_LEVELS = NUM_LEVELS

NY_TIMEZONE = "America/New_York"

DATA_CACHE_DIR = "data_cache"
SETTINGS_FILE = f"{DATA_CACHE_DIR}/app_settings.json"
REFRESH_STATUS_FILE = f"{DATA_CACHE_DIR}/refresh_status.json"