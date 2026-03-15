import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from options_dashboard.options_config import (
    DEFAULT_TICKERS,
    DEFAULT_EXPIRATION_WEIGHTS,
    DEFAULT_MAX_DISTANCE,
    DEFAULT_NUM_LEVELS,
    DATA_CACHE_DIR,
    SETTINGS_FILE,
    REFRESH_STATUS_FILE,
    NY_TIMEZONE,
)
from refresh_data import refresh_oi_data


st.set_page_config(page_title="OI Dashboard", layout="wide")
st.title("OI Dashboard")
st.caption("SPY and QQQ OI levels with scheduled refresh")

st_autorefresh(interval=60_000, key="dashboard_refresh")


def ensure_dirs():
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)


def load_json(path, fallback=None):
    if fallback is None:
        fallback = {}
    if not os.path.exists(path):
        return fallback
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return fallback


def save_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_settings():
    default_settings = {
        "tickers": DEFAULT_TICKERS,
        "weights": DEFAULT_EXPIRATION_WEIGHTS,
        "max_distance": DEFAULT_MAX_DISTANCE,
        "num_levels": DEFAULT_NUM_LEVELS,
    }
    saved = load_json(SETTINGS_FILE, default_settings)
    return {
        "tickers": saved.get("tickers", DEFAULT_TICKERS),
        "weights": saved.get("weights", DEFAULT_EXPIRATION_WEIGHTS),
        "max_distance": saved.get("max_distance", DEFAULT_MAX_DISTANCE),
        "num_levels": saved.get("num_levels", DEFAULT_NUM_LEVELS),
    }


def render_levels(title, payload):
    st.subheader(title)

    c1, c2, c3 = st.columns(3)
    c1.metric("Spot", payload.get("spot", "N/A"))
    c2.metric("Key Level", payload.get("key_level", "N/A"))
    c3.metric("Last Refresh (NY)", payload.get("refreshed_at_ny", "N/A"))

    st.write("**Expirations Used:**", payload.get("expirations_used", []))
    st.write("**Weights Used:**", payload.get("weights_used", []))
    st.write("**Search Range:**", payload.get("search_range", []))

    resistances = pd.DataFrame(payload.get("top_resistances", []))
    supports = pd.DataFrame(payload.get("top_supports", []))

    col_left, col_right = st.columns(2)

    with col_left:
        st.write("### Resistances")
        if resistances.empty:
            st.info("No resistance data.")
        else:
            if "weighted_open_interest" in resistances.columns:
                resistances = resistances.sort_values("weighted_open_interest", ascending=False)
            st.dataframe(resistances, use_container_width=True)

    with col_right:
        st.write("### Supports")
        if supports.empty:
            st.info("No support data.")
        else:
            if "weighted_open_interest" in supports.columns:
                supports = supports.sort_values("weighted_open_interest", ascending=False)
            st.dataframe(supports, use_container_width=True)


ensure_dirs()
settings = load_settings()

st.sidebar.header("Settings")

tickers = st.sidebar.multiselect(
    "Tickers",
    options=["SPY", "QQQ"],
    default=settings["tickers"],
)

weights_text = st.sidebar.text_input(
    "Expiration Weights",
    value=",".join(str(x) for x in settings["weights"]),
)

max_distance = st.sidebar.number_input(
    "Max Distance",
    min_value=1.0,
    max_value=100.0,
    value=float(settings["max_distance"]),
    step=1.0,
)

num_levels = st.sidebar.number_input(
    "Num Levels",
    min_value=1,
    max_value=10,
    value=int(settings["num_levels"]),
    step=1,
)

save_settings_btn = st.sidebar.button("Save Settings")
manual_refresh_btn = st.sidebar.button("Run Refresh Now")

try:
    weights = [float(x.strip()) for x in weights_text.split(",") if x.strip()]
except Exception:
    weights = DEFAULT_EXPIRATION_WEIGHTS
    st.sidebar.warning("Invalid weights format. Using defaults.")

if save_settings_btn:
    payload = {
        "tickers": tickers or DEFAULT_TICKERS,
        "weights": weights,
        "max_distance": max_distance,
        "num_levels": int(num_levels),
    }
    save_json(SETTINGS_FILE, payload)
    st.sidebar.success("Settings saved.")

if manual_refresh_btn:
    refresh_oi_data()
    st.sidebar.success("Refresh completed.")

status = load_json(REFRESH_STATUS_FILE, {})
st.sidebar.write("### Last Scheduled Refresh")
st.sidebar.write(status.get("last_refresh_ny", "No refresh yet"))

st.write("## Current Dashboard")

for ticker in tickers or DEFAULT_TICKERS:
    path = f"{DATA_CACHE_DIR}/oi_{ticker}.json"
    payload = load_json(path, {})
    if not payload:
        st.warning(f"No cached data yet for {ticker}. Run refresh first.")
    else:
        render_levels(f"{ticker} OI Levels", payload)
        st.divider()