import json
import os

import pandas as pd
import plotly.graph_objects as go
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
from options_common import get_intraday_history_last_24h_extended


if "POLYGON_API_KEY" in st.secrets and "POLYGON_API_KEY" not in os.environ:
    os.environ["POLYGON_API_KEY"] = st.secrets["POLYGON_API_KEY"]
if "MASSIVE_API_KEY" in st.secrets and "MASSIVE_API_KEY" not in os.environ:
    os.environ["MASSIVE_API_KEY"] = st.secrets["MASSIVE_API_KEY"]

st.set_page_config(page_title="Options Dashboard 4", layout="wide")
st.title("Options Dashboard 4")
st.caption("Polygon/Massive-based OI + Gamma + VEX dashboard")

# refresh every minute
st_autorefresh(interval=60000, key="dashboard_refresh")


@st.cache_data(ttl=60, show_spinner=False)
def cached_gamma(ticker, weights, max_distance, num_levels):
    return get_gamma_levels(
        ticker_symbol=ticker,
        weights=list(weights),
        max_distance=float(max_distance),
        num_levels=int(num_levels),
    )


@st.cache_data(ttl=60, show_spinner=False)
def cached_intraday_history(ticker):
    return get_intraday_history_last_24h_extended(ticker)


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


def get_regime_label(spot, gamma_flip, regime):
    if gamma_flip is None:
        if regime == "NO_LOCAL_FLIP_LONG_GAMMA_BIAS":
            return "LONG GAMMA BIAS (NO LOCAL FLIP)", "#1E88E5", None
        if regime == "NO_LOCAL_FLIP_SHORT_GAMMA_BIAS":
            return "SHORT GAMMA BIAS (NO LOCAL FLIP)", "#8E24AA", None
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


def get_gamma_mode_label(spot, gamma_flip, regime):
    if gamma_flip is None:
        if regime == "NO_LOCAL_FLIP_LONG_GAMMA_BIAS":
            return "LONG GAMMA PROXY MODE → FADE MOVES / BUY DIPS / SELL RIPS", "#1E88E5"
        if regime == "NO_LOCAL_FLIP_SHORT_GAMMA_BIAS":
            return "SHORT GAMMA PROXY MODE → FOLLOW MOMENTUM / BUY BREAKOUTS / SELL BREAKDOWNS", "#8E24AA"
        return "NO NEARBY FLIP → FOLLOW OVERALL BIAS", "#757575"

    if spot > gamma_flip:
        return "LONG GAMMA MODE → FADE MOVES / BUY DIPS / SELL RIPS", "#1E88E5"
    if spot < gamma_flip:
        return "SHORT GAMMA MODE → FOLLOW MOMENTUM / BUY BREAKOUTS / SELL BREAKDOWNS", "#8E24AA"
    return "AT FLIP → TRANSITION / REDUCE SIZE", "#FB8C00"


def render_oi_section(payload):
    st.subheader("Static OI Map")

    c1, c2, c3 = st.columns(3)
    c1.metric("OI Fixed Spot", round(float(payload.get("oi_fixed_spot", 0.0)), 2))
    c2.metric("OI Key Level", payload.get("key_level", "N/A"))
    c3.metric("OI Last Refresh (NY)", payload.get("refreshed_at_ny", "N/A"))

    if payload.get("refresh_mode"):
        st.write(f"**OI Refresh Mode:** {payload.get('refresh_mode')}")

    res = pd.DataFrame(payload.get("top_resistances", []))
    sup = pd.DataFrame(payload.get("top_supports", []))

    left, right = st.columns(2)
    with left:
        st.write("### OI Resistances")
        st.dataframe(res, use_container_width=True, hide_index=True)
    with right:
        st.write("### OI Supports")
        st.dataframe(sup, use_container_width=True, hide_index=True)


def render_gamma_section(gamma):
    st.subheader("Dynamic Gamma + VEX Map")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Spot", gamma.get("spot", "N/A"))
    c2.metric("Gamma Flip", gamma.get("gamma_flip", "None"))
    c3.metric("Gamma Key Level", gamma.get("key_level", "N/A"))
    c4.metric("Regime", gamma.get("regime", "N/A"))

    regime_label, regime_color, distance_pct = get_regime_label(
        gamma.get("spot"),
        gamma.get("gamma_flip"),
        gamma.get("regime"),
    )
    gamma_mode_label, gamma_mode_color = get_gamma_mode_label(
        gamma.get("spot"),
        gamma.get("gamma_flip"),
        gamma.get("regime"),
    )

    st.markdown(
        f"""
        <div style="
            background-color:{regime_color};
            padding:12px;
            border-radius:10px;
            color:white;
            font-weight:bold;
            font-size:22px;
            text-align:center;">
            REGIME STRENGTH: {regime_label}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style="
            background-color:{gamma_mode_color};
            padding:12px;
            border-radius:10px;
            color:white;
            font-weight:bold;
            font-size:18px;
            text-align:center;
            margin-top:8px;">
            GAMMA MODE: {gamma_mode_label}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if distance_pct is not None:
        st.write(f"**Distance from Flip:** {distance_pct:.2f}%")

    st.write(f"**Net Weighted GEX:** {gamma.get('total_net_gex', 0):,.0f}")
    st.write(f"**Flip Source:** {gamma.get('flip_source', 'unknown')}")

    left, right = st.columns(2)
    with left:
        st.write("### Gamma Resistances")
        st.dataframe(
            pd.DataFrame(gamma.get("top_resistances", [])),
            use_container_width=True,
            hide_index=True,
        )
        st.write("### VEX Resistances")
        st.dataframe(
            pd.DataFrame(gamma.get("top_vex_resistances", [])),
            use_container_width=True,
            hide_index=True,
        )
    with right:
        st.write("### Gamma Supports")
        st.dataframe(
            pd.DataFrame(gamma.get("top_supports", [])),
            use_container_width=True,
            hide_index=True,
        )
        st.write("### VEX Supports")
        st.dataframe(
            pd.DataFrame(gamma.get("top_vex_supports", [])),
            use_container_width=True,
            hide_index=True,
        )


def render_confluence_section(confluence):
    st.subheader("Confluence Engine")

    levels = confluence.get("levels", pd.DataFrame())
    if isinstance(levels, list):
        levels = pd.DataFrame(levels)

    if levels.empty:
        st.warning("No confluence levels found.")
        return

    cols = [
        "side",
        "level",
        "level_gex",
        "level_vex",
        "gamma_strength",
        "dynamic_score",
        "grade",
        "confidence",
        "hold_break_bias",
        "action",
        "distance_to_spot",
        "distance_to_key",
    ]
    cols = [c for c in cols if c in levels.columns]

    st.dataframe(levels[cols], use_container_width=True, hide_index=True)

    aplus = levels[levels["grade"] == "A+"]
    if not aplus.empty:
        st.success("A+ setups detected")
        st.dataframe(aplus[cols], use_container_width=True, hide_index=True)

    st.write("### When to Skip a Trade")
    for rule in confluence.get("skip_rules", []):
        st.write(f"- {rule}")


def get_chart_outcome_label(row):
    bias = str(row.get("hold_break_bias", ""))

    if bias == "LIKELY TO HOLD":
        return "BOUNCE"
    if bias == "CAN HOLD, BUT MESSY":
        return "MESSY BOUNCE"
    if bias == "LIKELY TO BREAK":
        return "BREAKTHROUGH"
    return "NEUTRAL"


def build_chart_for_ticker(ticker, hist_df, levels_df, current_spot):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=hist_df["datetime"],
            y=hist_df["close"],
            mode="lines",
            name=f"{ticker} Price",
        )
    )

    for _, row in levels_df.iterrows():
        level = float(row["level"])
        side = str(row.get("side", ""))
        action = str(row.get("action", "SKIP"))
        gex = row.get("level_gex", 0)
        vex = row.get("level_vex", 0)
        score = row.get("dynamic_score", 0)
        outcome = get_chart_outcome_label(row)

        line_color = "#00C853" if side == "SUPPORT" else "#D50000"
        dash = "solid" if side == "SUPPORT" else "dot"

        label_text = (
            f"{level:.2f} | Score {score} | "
            f"GEX {gex:,.0f} | VEX {vex:,.0f} | "
            f"{outcome} | {action}"
        )

        fig.add_hline(
            y=level,
            line_color=line_color,
            line_width=1.5,
            line_dash=dash,
            annotation_text=label_text,
            annotation_position="right",
        )

    fig.add_hline(
        y=float(current_spot),
        line_color="#1E88E5",
        line_width=1,
        line_dash="dash",
        annotation_text=f"Spot {current_spot:.2f}",
        annotation_position="left",
    )

    fig.update_layout(
        title=f"{ticker} - Last 24h (Premarket + Market + Aftermarket)",
        xaxis_title="Time",
        yaxis_title="Price",
        height=600,
        legend_title="Series",
        margin=dict(l=30, r=30, t=50, b=30),
    )
    return fig


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
    max_value=30,
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
    try:
        refresh_oi_data()
        st.sidebar.success("OI refresh completed.")
    except Exception as e:
        st.sidebar.error(f"OI refresh failed: {e}")

status = load_json(REFRESH_STATUS_FILE, {})
st.sidebar.write("### Last OI Refresh")
st.sidebar.write(status.get("last_refresh_ny", "No refresh yet"))

tab1, tab2 = st.tabs(["Dashboard", "Charts"])

ticker_data = {}

# Dashboard now uses gamma with extended-hours spot via options_common.get_current_spot_price()
for ticker in (tickers or DEFAULT_TICKERS):
    oi_path = os.path.join(DATA_CACHE_DIR, f"oi_{ticker}.json")
    oi_payload = load_json(oi_path, {})

    if not oi_payload:
        ticker_data[ticker] = {"error": "No OI cache found yet. Run the morning refresh first."}
        continue

    try:
        gamma = cached_gamma(
            ticker,
            tuple(weights),
            float(max_distance),
            int(num_levels),
        )

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

        ticker_data[ticker] = {
            "oi_payload": oi_payload,
            "gamma": gamma,
            "confluence": confluence,
        }
    except Exception as e:
        ticker_data[ticker] = {"error": str(e)}

with tab1:
    for ticker in (tickers or DEFAULT_TICKERS):
        st.header(ticker)
        data = ticker_data.get(ticker, {})
        if "error" in data:
            st.error(f"{ticker}: {data['error']}")
            st.divider()
            continue

        render_oi_section(data["oi_payload"])
        render_gamma_section(data["gamma"])
        render_confluence_section(data["confluence"])
        st.divider()

with tab2:
    st.write("Charts show the last 24 hours including premarket, market hours, and aftermarket.")

    for ticker in (tickers or DEFAULT_TICKERS):
        st.header(f"{ticker} Chart")
        data = ticker_data.get(ticker, {})
        if "error" in data:
            st.error(f"{ticker}: {data['error']}")
            st.divider()
            continue

        try:
            hist = cached_intraday_history(ticker)
            levels_df = data["confluence"]["levels"].copy()

            cols = [
                "side",
                "level",
                "level_gex",
                "level_vex",
                "dynamic_score",
                "hold_break_bias",
                "action",
                "grade",
                "gamma_strength",
            ]
            cols = [c for c in cols if c in levels_df.columns]

            fig = build_chart_for_ticker(
                ticker=ticker,
                hist_df=hist,
                levels_df=levels_df,
                current_spot=float(data["gamma"]["spot"]),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.write("### Level Summary")
            st.dataframe(levels_df[cols], use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"{ticker} chart error: {e}")

        st.divider()
