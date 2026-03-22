import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from options_config import (
    DEFAULT_TICKERS,
    DEFAULT_EXPIRATION_WEIGHTS,
    DEFAULT_MAX_DISTANCE,
    DEFAULT_NUM_LEVELS,
    DATA_CACHE_DIR,
    SETTINGS_FILE,
    REFRESH_STATUS_FILE,
    NY_TIMEZONE,
)
from oi_levels import get_oi_levels
from options_common import get_today_open_spot_price


def ensure_dirs():
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)


def load_settings():
    default_settings = {
        "tickers": DEFAULT_TICKERS,
        "weights": DEFAULT_EXPIRATION_WEIGHTS,
        "max_distance": DEFAULT_MAX_DISTANCE,
        "num_levels": DEFAULT_NUM_LEVELS,
    }

    if not os.path.exists(SETTINGS_FILE):
        return default_settings

    try:
        with open(SETTINGS_FILE, "r") as f:
            saved = json.load(f)
        return {
            "tickers": saved.get("tickers", DEFAULT_TICKERS),
            "weights": saved.get("weights", DEFAULT_EXPIRATION_WEIGHTS),
            "max_distance": saved.get("max_distance", DEFAULT_MAX_DISTANCE),
            "num_levels": saved.get("num_levels", DEFAULT_NUM_LEVELS),
        }
    except Exception:
        return default_settings


def dataframe_to_records(df):
    if df is None or len(df) == 0:
        return []
    return df.to_dict(orient="records")


def save_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def refresh_oi_data():
    ensure_dirs()
    settings = load_settings()
    now_ny = datetime.now(ZoneInfo(NY_TIMEZONE))

    tickers = settings["tickers"]
    weights = settings["weights"]
    max_distance = settings["max_distance"]
    num_levels = settings["num_levels"]

    saved_files = []

    for ticker in tickers:
        open_spot = get_today_open_spot_price(ticker)

        result = get_oi_levels(
            ticker_symbol=ticker,
            weights=weights,
            max_distance=max_distance,
            num_levels=num_levels,
            fixed_spot=open_spot,
        )

        payload = {
            "model": result["model"],
            "ticker": result["ticker"],
            "spot": result["spot"],
            "oi_fixed_spot": open_spot,
            "expirations_used": result["expirations_used"],
            "weights_used": result["weights_used"],
            "search_range": result["search_range"],
            "key_level": result["key_level"],
            "top_resistances": dataframe_to_records(result["top_resistances"]),
            "top_supports": dataframe_to_records(result["top_supports"]),
            "refreshed_at_ny": now_ny.isoformat(),
        }

        output_file = os.path.join(DATA_CACHE_DIR, f"oi_{ticker}.json")
        save_json(output_file, payload)
        saved_files.append(output_file)

    refresh_status = {
        "last_refresh_ny": now_ny.isoformat(),
        "tickers": tickers,
        "weights": weights,
        "max_distance": max_distance,
        "num_levels": num_levels,
        "files_written": saved_files,
    }

    save_json(REFRESH_STATUS_FILE, refresh_status)


if __name__ == "__main__":
    refresh_oi_data()