import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

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
from refresh_data import refresh_oi_data
from gamma_exposure import get_gamma_levels
from confluence_levels import build_confluence_levels


st.set_page_config(page_title="Options Dashboard", layout="wide")
st.title("Options Dashboard")
st.caption("Static OI from 9:30 AM NY open + dynamic Gamma + confluence scoring")

st_autorefresh(interval=300000, key="dashboard_refresh") # 5 minutes


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


def should_force_refresh(status: dict) -> bool:
    now_ny = datetime.now(ZoneInfo(NY_TIMEZONE))
    if now_ny.weekday() >= 5:
        return False
    if (now_ny.hour, now_ny.minute) < (9, 30):
        return False

    last_refresh = status.get("last_refresh_ny")
    if not last_refresh:
        return True

    try:
        last_dt = datetime.fromisoformat(last_refresh)
    except Exception:
        return True

    return last_dt.date() != now_ny.date()


def render_oi_section(payload):
    st.subheader("Static OI Map")
    c1, c2, c3 = st.columns(3)
    c1.metric("OI Fixed Spot", payload.get("oi_fixed_spot", "N/A"))
    c2.metric("OI Key Level", payload.get("key_level", "N/A"))
    c3.metric("OI Last Refresh (NY)", payload.get("refreshed_at_ny", "N/A"))

    st.write("**Expirations Used:**", payload.get("expirations_used", []))
    st.write("**Weights Used:**", payload.get("weights_used", []))

    res = pd.DataFrame(payload.get("top_resistances", []))
    sup = pd.DataFrame(payload.get("top_supports", []))

    left, right = st.columns(2)

    with left:
        st.write("### OI Resistances")
        if not res.empty and "weighted_open_interest" in res.columns:
            res = res.sort_values("weighted_open_interest", ascending=False)
        st.dataframe(res, use_container_width=True)

    with right:
        st.write("### OI Supports")
        if not sup.empty and "weighted_open_interest" in sup.columns:
            sup = sup.sort_values("weighted_open_interest", ascending=False)
        st.dataframe(sup, use_container_width=True)


def render_gamma_section(gamma):
    st.subheader("Dynamic Gamma Map")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Spot", gamma.get("spot", "N/A"))
    c2.metric("Gamma Flip", gamma.get("gamma_flip", "None"))
    c3.metric("Gamma Key Level", gamma.get("key_level", "N/A"))
    c4.metric("Regime", gamma.get("regime", "N/A"))

    res = gamma["top_resistances"] if isinstance(gamma["top_resistances"], pd.DataFrame) else pd.DataFrame(gamma["top_resistances"])
    sup = gamma["top_supports"] if isinstance(gamma["top_supports"], pd.DataFrame) else pd.DataFrame(gamma["top_supports"])

    left, right = st.columns(2)

    with left:
        st.write("### Gamma Resistances")
        if not res.empty and "weighted_gex" in res.columns:
            res = res.sort_values("weighted_gex", ascending=False)
        st.dataframe(res, use_container_width=True)

    with right:
        st.write("### Gamma Supports")
        if not sup.empty and "weighted_gex" in sup.columns:
            sup = sup.sort_values("weighted_gex", ascending=True)
        st.dataframe(sup, use_container_width=True)


def render_confluence_section(confluence):
    st.subheader("Confluence Engine")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Spot", confluence.get("spot", "N/A"))
    c2.metric("Gamma Flip", confluence.get("gamma_flip", "None"))
    c3.metric("OI Key", confluence.get("oi_key_level", "N/A"))
    c4.metric("Gamma Key", confluence.get("gamma_key_level", "N/A"))

    st.write("**Regime:**", confluence.get("regime", "N/A"))

    levels = confluence.get("levels", pd.DataFrame())
    if isinstance(levels, list):
        levels = pd.DataFrame(levels)

    if not levels.empty:
        st.write("### Trade Quality Levels")
        st.dataframe(levels, use_container_width=True)

        aplus = levels[levels["grade"] == "A+"]
        if not aplus.empty:
            st.success("A+ setups detected")
            st.dataframe(aplus, use_container_width=True)
        else:
            st.info("No A+ setups right now.")
    else:
        st.warning("No confluence levels found.")

    st.write("### When to Skip a Trade")
    for rule in confluence.get("skip_rules", []):
        st.write(f"- {rule}")


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
    max_value=100,
    value=int(settings["num_levels"]),
    step=1,
)

save_settings_btn = st.sidebar.button("Save Settings")
manual_refresh_btn = st.sidebar.button("Run OI Refresh Now")

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
    st.sidebar.success("OI refresh completed.")

status = load_json(REFRESH_STATUS_FILE, {})
if should_force_refresh(status):
    refresh_oi_data()
    status = load_json(REFRESH_STATUS_FILE, {})

st.sidebar.write("### Last OI Refresh")
st.sidebar.write(status.get("last_refresh_ny", "No refresh yet"))

for ticker in tickers or DEFAULT_TICKERS:
    st.header(f"{ticker}")

    oi_path = os.path.join(DATA_CACHE_DIR, f"oi_{ticker}.json")
    oi_payload = load_json(oi_path, {})

    if not oi_payload:
        st.warning(f"No OI cache found yet for {ticker}.")
        continue

    render_oi_section(oi_payload)

    gamma = get_gamma_levels(
        ticker_symbol=ticker,
        weights=weights,
        max_distance=max_distance,
        num_levels=int(num_levels),
    )
    render_gamma_section(gamma)

    confluence = build_confluence_levels(
        ticker_symbol=ticker,
        oi_fixed_spot=oi_payload.get("oi_fixed_spot"),
        weights=weights,
        max_distance=max_distance,
        num_levels=int(num_levels),
    )
    render_confluence_section(confluence)

    st.divider()