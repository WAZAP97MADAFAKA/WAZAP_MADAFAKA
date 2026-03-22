import json
import os

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
)
from refresh_data import refresh_oi_data
from gamma_exposure import get_gamma_levels
from confluence_levels import build_confluence_from_results


st.set_page_config(page_title="Options Dashboard", layout="wide")
st.title("Options Dashboard")
st.caption("Static OI from 9:30 AM NY open + dynamic Gamma + confluence scoring")

st_autorefresh(interval=86400000, key="dashboard_refresh")


@st.cache_data(ttl=600, show_spinner=False)
def cached_gamma(ticker, weights, max_distance, num_levels):
    return get_gamma_levels(
        ticker_symbol=ticker,
        weights=weights,
        max_distance=max_distance,
        num_levels=num_levels,
    )


def get_regime_label(spot, gamma_flip):
    if gamma_flip is None:
        return "NO FLIP DETECTED", "#9E9E9E", None

    distance_pct = abs(spot - gamma_flip) / spot * 100

    if distance_pct < 0.5:
        return "CHOP / TRANSITION", "#FFA726", distance_pct

    if distance_pct < 1.5:
        if spot > gamma_flip:
            return "MODERATE BULLISH", "#66BB6A", distance_pct
        return "MODERATE BEARISH", "#EF5350", distance_pct

    if spot > gamma_flip:
        return "STRONG BULLISH", "#00C853", distance_pct
    return "STRONG BEARISH", "#D50000", distance_pct


def get_gamma_mode_label(spot, gamma_flip):
    if gamma_flip is None:
        return "NO NEARBY FLIP → FOLLOW OVERALL BIAS", "#757575"

    if spot > gamma_flip:
        return "LONG GAMMA MODE → FADE MOVES / BUY DIPS / SELL RIPS", "#1E88E5"

    if spot < gamma_flip:
        return "SHORT GAMMA MODE → FOLLOW MOMENTUM / BUY BREAKOUTS / SELL BREAKDOWNS", "#8E24AA"

    return "AT FLIP → TRANSITION / REDUCE SIZE", "#FB8C00"


def gamma_strength_color(value):
    mapping = {
        "STRONG_GAMMA_BACKED": "#1B5E20",
        "STRONG_BUT_VOLATILE": "#1565C0",
        "GAMMA_BACKED": "#2E7D32",
        "WEAK_GAMMA_SUPPORT": "#F57C00",
        "WEAK_GAMMA_RESISTANCE": "#E65100",
        "NO_GAMMA_BACKING": "#B71C1C",
    }
    return mapping.get(value, "#616161")


def grade_color(value):
    mapping = {
        "A+": "#00C853",
        "A": "#64DD17",
        "B": "#FFD600",
        "SKIP": "#D50000",
    }
    return mapping.get(value, "#616161")


def build_level_strength_card(row):
    grade = row.get("grade", "")
    gamma_strength = row.get("gamma_strength", "")
    side = row.get("side", "")
    level = row.get("level", "")

    if gamma_strength == "STRONG_GAMMA_BACKED":
        return f"{grade} | {side} {level} | VERY STRONG"
    if gamma_strength == "STRONG_BUT_VOLATILE":
        return f"{grade} | {side} {level} | STRONG BUT VOLATILE"
    if gamma_strength == "GAMMA_BACKED":
        return f"{grade} | {side} {level} | BACKED"
    if gamma_strength == "WEAK_GAMMA_SUPPORT":
        return f"{grade} | {side} {level} | WEAK SUPPORT"
    if gamma_strength == "WEAK_GAMMA_RESISTANCE":
        return f"{grade} | {side} {level} | WEAK RESISTANCE"
    if gamma_strength == "NO_GAMMA_BACKING":
        return f"{grade} | {side} {level} | LIKELY FAIL"
    return f"{grade} | {side} {level}"


def style_trade_quality_table(df: pd.DataFrame):
    def style_grade(val):
        color = grade_color(val)
        return f"background-color: {color}; color: white; font-weight: bold;"

    def style_gamma_strength(val):
        color = gamma_strength_color(val)
        return f"background-color: {color}; color: white; font-weight: bold;"

    def style_level_strength_card(val):
        text = str(val)
        if "VERY STRONG" in text:
            return "background-color: #1B5E20; color: white; font-weight: bold;"
        if "STRONG BUT VOLATILE" in text:
            return "background-color: #1565C0; color: white; font-weight: bold;"
        if "BACKED" in text:
            return "background-color: #2E7D32; color: white; font-weight: bold;"
        if "WEAK SUPPORT" in text:
            return "background-color: #F57C00; color: white; font-weight: bold;"
        if "WEAK RESISTANCE" in text:
            return "background-color: #E65100; color: white; font-weight: bold;"
        if "LIKELY FAIL" in text:
            return "background-color: #B71C1C; color: white; font-weight: bold;"
        return ""

    styled = (
        df.style
        .map(style_grade, subset=["grade"])
        .map(style_gamma_strength, subset=["gamma_strength"])
        .map(style_level_strength_card, subset=["level_strength_card"])
    )
    return styled


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

    regime_label, regime_color, distance_pct = get_regime_label(
        gamma.get("spot"),
        gamma.get("gamma_flip"),
    )

    gamma_mode_label, gamma_mode_color = get_gamma_mode_label(
        gamma.get("spot"),
        gamma.get("gamma_flip"),
    )

    st.markdown(
        f"""
        <div style="
            background-color:{regime_color};
            padding:12px;
            border-radius:10px;
            margin-bottom:12px;
            color:white;
            font-weight:bold;
            font-size:22px;
            text-align:center;">
            REGIME STRENGTH: {regime_label}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="
            background-color:{gamma_mode_color};
            padding:12px;
            border-radius:10px;
            margin-bottom:12px;
            color:white;
            font-weight:bold;
            font-size:20px;
            text-align:center;">
            GAMMA MODE: {gamma_mode_label}
        </div>
        """,
        unsafe_allow_html=True
    )

    if distance_pct is not None:
        st.write(f"**Distance from Flip:** {distance_pct:.2f}%")

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

        display_levels = levels.copy()
        display_levels["level_strength_card"] = display_levels.apply(build_level_strength_card, axis=1)

        column_order = [
            "side",
            "level",
            "gamma_match",
            "gamma_strength",
            "grade",
            "level_strength_card",
        ]
        column_order = [col for col in column_order if col in display_levels.columns]
        display_levels = display_levels[column_order]

        st.dataframe(
            style_trade_quality_table(display_levels),
            use_container_width=True,
            hide_index=True,
        )

        aplus = levels[levels["grade"] == "A+"]
        if not aplus.empty:
            st.success("A+ setups detected")
            st.dataframe(aplus, use_container_width=True, hide_index=True)
        else:
            st.info("No A+ setups right now.")

        weak_levels = levels[
            levels["gamma_strength"].isin(["NO_GAMMA_BACKING", "WEAK_GAMMA_SUPPORT", "WEAK_GAMMA_RESISTANCE"])
        ]
        if not weak_levels.empty:
            st.warning("Weak levels detected — these are more likely to fail.")
            st.dataframe(weak_levels, use_container_width=True, hide_index=True)

        volatile_levels = levels[levels["gamma_strength"] == "STRONG_BUT_VOLATILE"]
        if not volatile_levels.empty:
            st.info("Strong but volatile levels detected — these can react hard but may not be clean.")
            st.dataframe(volatile_levels, use_container_width=True, hide_index=True)
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
    default=settings["tickers"] if settings["tickers"] else ["SPY", "QQQ"],
)

weights_text = st.sidebar.text_input(
    "Expiration Weights",
    value=",".join(str(x) for x in settings["weights"]),
)

max_distance_value = float(settings["max_distance"])
max_distance_value = max(1.0, min(100.0, max_distance_value))

max_distance = st.sidebar.number_input(
    "Max Distance",
    min_value=1.0,
    max_value=100.0,
    value=max_distance_value,
    step=1.0,
)

num_levels_value = int(settings["num_levels"])
num_levels_value = max(1, min(100, num_levels_value))

num_levels = st.sidebar.number_input(
    "Num Levels",
    min_value=1,
    max_value=100,
    value=num_levels_value,
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
    try:
        refresh_oi_data()
        st.sidebar.success("OI refresh completed.")
    except Exception as e:
        st.sidebar.error(f"OI refresh failed: {e}")

status = load_json(REFRESH_STATUS_FILE, {})
st.sidebar.write("### Last OI Refresh")
st.sidebar.write(status.get("last_refresh_ny", "No refresh yet"))

for ticker in (tickers or DEFAULT_TICKERS):
    st.header(f"{ticker}")

    oi_path = os.path.join(DATA_CACHE_DIR, f"oi_{ticker}.json")
    oi_payload = load_json(oi_path, {})

    if not oi_payload:
        st.warning(f"No OI cache found yet for {ticker}. Run the morning refresh first.")
        st.divider()
        continue

    render_oi_section(oi_payload)

    try:
        gamma = cached_gamma(
            ticker,
            tuple(weights),
            float(max_distance),
            int(num_levels),
        )
        render_gamma_section(gamma)

        oi_for_confluence = {
            "key_level": oi_payload.get("key_level"),
            "top_resistances": pd.DataFrame(oi_payload.get("top_resistances", [])),
            "top_supports": pd.DataFrame(oi_payload.get("top_supports", [])),
            "spot": oi_payload.get("oi_fixed_spot"),
        }

        confluence = build_confluence_from_results(
            ticker_symbol=ticker,
            oi=oi_for_confluence,
            gamma=gamma,
        )
        render_confluence_section(confluence)

    except Exception as e:
        st.error(f"{ticker} gamma/confluence error: {e}")

    st.divider()