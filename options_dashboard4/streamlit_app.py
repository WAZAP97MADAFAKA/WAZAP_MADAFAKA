import json
import os
from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
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


def slice_history_last_hours(hist_df: pd.DataFrame, hours: int) -> pd.DataFrame:
    if hist_df.empty:
        return hist_df

    df = hist_df.copy()
    if "datetime" not in df.columns:
        return df

    df = df.sort_values("datetime").reset_index(drop=True)
    end_time = df["datetime"].max()
    start_time = end_time - pd.Timedelta(hours=hours)
    df = df[df["datetime"] >= start_time].copy()
    return df.reset_index(drop=True)


def get_aligned_y_range(hist_df: pd.DataFrame, levels_df: pd.DataFrame, current_spot: float):
    values = []

    if hist_df is not None and not hist_df.empty and "close" in hist_df.columns:
        values.extend(pd.to_numeric(hist_df["close"], errors="coerce").dropna().tolist())

    if levels_df is not None and not levels_df.empty and "level" in levels_df.columns:
        values.extend(pd.to_numeric(levels_df["level"], errors="coerce").dropna().tolist())

    if current_spot is not None:
        values.append(float(current_spot))

    if not values:
        return None

    y_min = min(values)
    y_max = max(values)
    spread = y_max - y_min

    if spread <= 0:
        pad = max(abs(y_max) * 0.01, 1.0)
    else:
        pad = spread * 0.08

    return [round(y_min - pad, 2), round(y_max + pad, 2)]


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

    st.write("### Decision Table")

    cols = [
        "side",
        "level",
        "level_gex",
        "level_vex",
        "vex_strength",
        "gamma_strength",
        "market_behavior",
        "best_trade_type",
        "direction",
        "static_score",
        "static_grade",
        "dynamic_score",
        "grade",
        "confidence",
        "hold_break_bias",
        "trade_now_signal",
        "bounce_probability",
        "breakout_probability",
        "entry",
        "stop",
        "target",
        "distance_to_spot",
    ]
    cols = [c for c in cols if c in levels.columns]

    st.dataframe(levels[cols], use_container_width=True, hide_index=True)

    if "trade_now_signal" in levels.columns:
        active = levels[levels["trade_now_signal"] == "TRADE NOW"]
        if not active.empty:
            st.success("TRADE NOW setups detected")
            st.dataframe(active[cols], use_container_width=True, hide_index=True)

        watch = levels[levels["trade_now_signal"] == "WATCH"]
        if not watch.empty:
            st.info("WATCH setups detected")
            st.dataframe(watch[cols], use_container_width=True, hide_index=True)

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


def add_session_backgrounds(fig, hist_df):
    if hist_df.empty:
        return fig

    x_min = hist_df["datetime"].min()
    x_max = hist_df["datetime"].max()

    start_day = x_min.normalize()
    end_day = x_max.normalize()

    current_day = start_day
    while current_day <= end_day:
        if current_day.weekday() < 5:
            overnight_start = current_day
            overnight_end = current_day + timedelta(hours=4)

            premarket_start = current_day + timedelta(hours=4)
            premarket_end = current_day + timedelta(hours=9, minutes=30)

            aftermarket_start = current_day + timedelta(hours=16)
            aftermarket_end = current_day + timedelta(hours=20)

            overnight2_start = current_day + timedelta(hours=20)
            overnight2_end = current_day + timedelta(days=1)

            windows = [
                (overnight_start, overnight_end, "rgba(80, 80, 120, 0.12)"),
                (premarket_start, premarket_end, "rgba(70, 120, 180, 0.12)"),
                (aftermarket_start, aftermarket_end, "rgba(150, 90, 170, 0.12)"),
                (overnight2_start, overnight2_end, "rgba(80, 80, 120, 0.12)"),
            ]

            for x0, x1, color in windows:
                left = max(x0, x_min)
                right = min(x1, x_max)
                if left < right:
                    fig.add_vrect(
                        x0=left,
                        x1=right,
                        fillcolor=color,
                        opacity=1,
                        line_width=0,
                        layer="below",
                    )

        current_day += timedelta(days=1)

    return fig

def build_hybrid_subplot_figure(
    ticker,
    hist_df,
    levels_df,
    gamma,
    oi_key_level,
    forced_y_range=None,
):
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.03,
        column_widths=[0.62, 0.38],
    )

    # -----------------------------
    # Left side: price + OI levels
    # -----------------------------
    fig.add_trace(
        go.Scatter(
            x=hist_df["datetime"],
            y=hist_df["close"],
            mode="lines",
            name=f"{ticker} Price",
            line=dict(width=2),
        ),
        row=1,
        col=1,
    )

    # Session backgrounds on left subplot only
    if not hist_df.empty:
        x_min = hist_df["datetime"].min()
        x_max = hist_df["datetime"].max()
        start_day = x_min.normalize()
        end_day = x_max.normalize()

        current_day = start_day
        while current_day <= end_day:
            if current_day.weekday() < 5:
                overnight_start = current_day
                overnight_end = current_day + timedelta(hours=4)

                premarket_start = current_day + timedelta(hours=4)
                premarket_end = current_day + timedelta(hours=9, minutes=30)

                aftermarket_start = current_day + timedelta(hours=16)
                aftermarket_end = current_day + timedelta(hours=20)

                overnight2_start = current_day + timedelta(hours=20)
                overnight2_end = current_day + timedelta(days=1)

                windows = [
                    (overnight_start, overnight_end, "rgba(80, 80, 120, 0.12)"),
                    (premarket_start, premarket_end, "rgba(70, 120, 180, 0.12)"),
                    (aftermarket_start, aftermarket_end, "rgba(150, 90, 170, 0.12)"),
                    (overnight2_start, overnight2_end, "rgba(80, 80, 120, 0.12)"),
                ]

                for x0, x1, color in windows:
                    left = max(x0, x_min)
                    right = min(x1, x_max)
                    if left < right:
                        fig.add_vrect(
                            x0=left,
                            x1=right,
                            fillcolor=color,
                            opacity=1,
                            line_width=0,
                            layer="below",
                            row=1,
                            col=1,
                        )

            current_day += timedelta(days=1)

    for _, row in levels_df.iterrows():
        level = float(row["level"])
        side = str(row.get("side", ""))
        line_color = "#00C853" if side == "SUPPORT" else "#D50000"
        dash = "solid" if side == "SUPPORT" else "dot"

        fig.add_hline(
            y=level,
            line_color=line_color,
            line_width=1.4,
            line_dash=dash,
            row=1,
            col=1,
        )

        fig.add_annotation(
            xref="paper",
            yref="y",
            x=0.60,
            y=level,
            text=f"{level:.2f}",
            showarrow=False,
            font=dict(size=10, color=line_color),
            bgcolor="rgba(0,0,0,0.25)",
            xanchor="left",
            yanchor="middle",
        )

    current_spot = float(gamma["spot"])
    fig.add_hline(
        y=current_spot,
        line_color="#64B5F6",
        line_width=1.4,
        line_dash="dash",
        row=1,
        col=1,
    )

    fig.add_annotation(
        xref="paper",
        yref="y",
        x=0.01,
        y=current_spot,
        text=f"Spot {current_spot:.2f}",
        showarrow=False,
        font=dict(size=11, color="#64B5F6"),
        bgcolor="rgba(0,0,0,0.35)",
        xanchor="left",
        yanchor="middle",
    )

    # -----------------------------
    # Right side: GEX bars
    # -----------------------------
    curve = pd.DataFrame(gamma.get("gex_curve", []))
    if curve.empty:
        curve = pd.DataFrame(gamma.get("gex_curve_wide", []))

    if curve.empty:
        return fig, pd.DataFrame()

    curve = curve.sort_values("strike").reset_index(drop=True)

    support_strikes = set()
    resistance_strikes = set()

    top_supports_df = pd.DataFrame(gamma.get("top_supports", []))
    top_resistances_df = pd.DataFrame(gamma.get("top_resistances", []))

    if not top_supports_df.empty and "strike" in top_supports_df.columns:
        support_strikes = set(top_supports_df["strike"].astype(float).tolist())

    if not top_resistances_df.empty and "strike" in top_resistances_df.columns:
        resistance_strikes = set(top_resistances_df["strike"].astype(float).tolist())

    def classify_gamma_side(strike):
        strike = float(strike)
        if strike in support_strikes:
            return "SUPPORT"
        if strike in resistance_strikes:
            return "RESISTANCE"
        return "OTHER"

    curve["gamma_side"] = curve["strike"].apply(classify_gamma_side)
    curve["abs_weighted_gex"] = curve["weighted_gex"].abs()

    colors = ["#00C853" if float(v) >= 0 else "#D50000" for v in curve["weighted_gex"]]

    fig.add_trace(
        go.Bar(
            x=curve["weighted_gex"],
            y=curve["strike"],
            orientation="h",
            marker_color=colors,
            name="Weighted GEX",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    gamma_key_level = gamma.get("key_level")
    gamma_flip = gamma.get("gamma_flip")

    if gamma_key_level is not None:
        fig.add_hline(
            y=float(gamma_key_level),
            line_width=2,
            line_dash="dash",
            line_color="#FFD54F",
            row=1,
            col=2,
        )
        fig.add_annotation(
            xref="paper",
            yref="y",
            x=0.985,
            y=float(gamma_key_level),
            text=f"Gamma Key {float(gamma_key_level):.2f}",
            showarrow=False,
            font=dict(color="#FFD54F", size=11),
            bgcolor="rgba(0,0,0,0.35)",
            xanchor="right",
            yanchor="bottom",
        )

    if oi_key_level is not None:
        fig.add_hline(
            y=float(oi_key_level),
            line_width=2,
            line_dash="dot",
            line_color="#64B5F6",
            row=1,
            col=2,
        )
        fig.add_annotation(
            xref="paper",
            yref="y",
            x=0.76,
            y=float(oi_key_level),
            text=f"OI Key {float(oi_key_level):.2f}",
            showarrow=False,
            font=dict(color="#64B5F6", size=11),
            bgcolor="rgba(0,0,0,0.35)",
            xanchor="left",
            yanchor="bottom",
        )

    if gamma_flip is not None:
        fig.add_hline(
            y=float(gamma_flip),
            line_width=2,
            line_dash="longdash",
            line_color="#FF9800",
            row=1,
            col=2,
        )
        fig.add_annotation(
            xref="paper",
            yref="y",
            x=0.985,
            y=float(gamma_flip),
            text=f"Gamma Flip {float(gamma_flip):.2f}",
            showarrow=False,
            font=dict(color="#FF9800", size=11),
            bgcolor="rgba(0,0,0,0.35)",
            xanchor="right",
            yanchor="top",
        )

    max_abs_x = max(curve["abs_weighted_gex"].max(), 1.0)

    for _, row in curve.iterrows():
        strike = float(row["strike"])
        gex_val = float(row["weighted_gex"])
        gamma_side = row["gamma_side"]

        if gamma_side == "SUPPORT":
            txt = "S"
            color = "#00E676"
        elif gamma_side == "RESISTANCE":
            txt = "R"
            color = "#FF5252"
        else:
            continue

        label_x = gex_val + (0.03 * max_abs_x if gex_val >= 0 else -0.03 * max_abs_x)
        fig.add_annotation(
            x=label_x,
            y=strike,
            xref="x2",
            yref="y",
            text=txt,
            showarrow=False,
            font=dict(color=color, size=11),
            bgcolor="rgba(0,0,0,0.25)",
            xanchor="center",
            yanchor="middle",
        )

    shared_yaxis = get_shared_yaxis_config(forced_y_range)

    fig.update_yaxes(shared_yaxis, row=1, col=1)
    fig.update_yaxes(shared_yaxis, row=1, col=2)

    fig.update_xaxes(
        title_text="Time",
        rangebreaks=[dict(bounds=["sat", "mon"])],
        rangeslider=dict(visible=False),
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text="Weighted GEX",
        row=1,
        col=2,
    )

    fig.update_layout(
        title=f"{ticker} - Hybrid View",
        template="plotly_dark",
        height=700,
        margin=dict(l=60, r=100, t=70, b=50),
        showlegend=False,
    )

    return fig, curve


def get_shared_yaxis_config(forced_y_range):
    if forced_y_range is None:
        return {}

    y_min, y_max = forced_y_range
    return {
        "range": forced_y_range,
        "tickmode": "linear",
        "tick0": int(y_min // 5) * 5,
        "dtick": 5,
        "fixedrange": True,
    }

def get_shared_yaxis_config(forced_y_range):
    if forced_y_range is None:
        return {}

    y_min, y_max = forced_y_range

    # Use 5-point spacing so OI lines and strike bars line up visually
    return {
        "range": forced_y_range,
        "tickmode": "linear",
        "tick0": int(y_min // 5) * 5,
        "dtick": 5,
        "fixedrange": True,
    }

def build_chart_for_ticker(
    ticker,
    hist_df,
    levels_df,
    current_spot,
    forced_y_range=None,
    title_suffix="Last 24h (Premarket + Market + Aftermarket + Overnight)",
):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=hist_df["datetime"],
            y=hist_df["close"],
            mode="lines",
            name=f"{ticker} Price",
        )
    )

    fig = add_session_backgrounds(fig, hist_df)

    for _, row in levels_df.iterrows():
        level = float(row["level"])
        side = str(row.get("side", ""))
        direction = str(row.get("direction", "SKIP"))
        gex = row.get("level_gex", 0)
        vex = row.get("level_vex", 0)
        dynamic_score = row.get("dynamic_score", 0)
        static_score = row.get("static_score", 0)
        signal = row.get("trade_now_signal", "PLAN")
        behavior = row.get("market_behavior", "")
        outcome = get_chart_outcome_label(row)

        line_color = "#00C853" if side == "SUPPORT" else "#D50000"
        dash = "solid" if side == "SUPPORT" else "dot"

        label_text = (
            f"{level:.2f} | Static {static_score} | Dynamic {dynamic_score} | "
            f"{behavior} | {signal} | GEX {gex:,.0f} | VEX {vex:,.0f} | "
            f"{outcome} | {direction}"
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

    shared_yaxis = get_shared_yaxis_config(forced_y_range)

    fig.update_layout(
        title=f"{ticker} - {title_suffix}",
        xaxis_title="Time",
        yaxis_title="Price / Strike",
        height=650,
        legend_title="Series",
        margin=dict(l=70, r=70, t=70, b=50),
        xaxis=dict(
            rangebreaks=[dict(bounds=["sat", "mon"])],
            rangeslider=dict(visible=False),
        ),
        yaxis=shared_yaxis,
    )

    return fig


def build_hybrid_gex_chart(ticker, gamma, oi_key_level, forced_y_range=None):
    curve = pd.DataFrame(gamma.get("gex_curve", []))
    if curve.empty:
        curve = pd.DataFrame(gamma.get("gex_curve_wide", []))

    if curve.empty:
        return None, pd.DataFrame()

    curve = curve.sort_values("strike").reset_index(drop=True)

    support_strikes = set()
    resistance_strikes = set()

    top_supports_df = pd.DataFrame(gamma.get("top_supports", []))
    top_resistances_df = pd.DataFrame(gamma.get("top_resistances", []))

    if not top_supports_df.empty and "strike" in top_supports_df.columns:
        support_strikes = set(top_supports_df["strike"].astype(float).tolist())

    if not top_resistances_df.empty and "strike" in top_resistances_df.columns:
        resistance_strikes = set(top_resistances_df["strike"].astype(float).tolist())

    def classify_gamma_side(strike):
        strike = float(strike)
        if strike in support_strikes:
            return "SUPPORT"
        if strike in resistance_strikes:
            return "RESISTANCE"
        return "OTHER"

    curve["gamma_side"] = curve["strike"].apply(classify_gamma_side)
    curve["abs_weighted_gex"] = curve["weighted_gex"].abs()

    colors = ["#00C853" if float(v) >= 0 else "#D50000" for v in curve["weighted_gex"]]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=curve["strike"],
            x=curve["weighted_gex"],
            orientation="h",
            marker_color=colors,
            name="Weighted GEX",
        )
    )

    gamma_key_level = gamma.get("key_level")
    gamma_flip = gamma.get("gamma_flip")
    current_spot = gamma.get("spot")

    if gamma_key_level is not None:
        fig.add_hline(
            y=float(gamma_key_level),
            line_width=2,
            line_dash="dash",
            line_color="#FFD54F",
        )
        fig.add_annotation(
            xref="paper",
            yref="y",
            x=0.98,
            y=float(gamma_key_level),
            text=f"Gamma Key {float(gamma_key_level):.2f}",
            showarrow=False,
            font=dict(color="#FFD54F"),
            bgcolor="rgba(0,0,0,0.35)",
            xanchor="right",
            yanchor="bottom",
            xshift=-4,
        )

    if oi_key_level is not None:
        fig.add_hline(
            y=float(oi_key_level),
            line_width=2,
            line_dash="dot",
            line_color="#64B5F6",
        )
        fig.add_annotation(
            xref="paper",
            yref="y",
            x=0.02,
            y=float(oi_key_level),
            text=f"OI Key {float(oi_key_level):.2f}",
            showarrow=False,
            font=dict(color="#64B5F6"),
            bgcolor="rgba(0,0,0,0.35)",
            xanchor="left",
            yanchor="bottom",
            xshift=4,
        )

    if gamma_flip is not None:
        fig.add_hline(
            y=float(gamma_flip),
            line_width=2,
            line_dash="longdash",
            line_color="#FF9800",
        )
        fig.add_annotation(
            xref="paper",
            yref="y",
            x=0.98,
            y=float(gamma_flip),
            text=f"Gamma Flip {float(gamma_flip):.2f}",
            showarrow=False,
            font=dict(color="#FF9800"),
            bgcolor="rgba(0,0,0,0.35)",
            xanchor="right",
            yanchor="top",
            xshift=-4,
        )

    if current_spot is not None:
        fig.add_hline(
            y=float(current_spot),
            line_width=2,
            line_dash="solid",
            line_color="#FFFFFF",
        )
        fig.add_annotation(
            xref="paper",
            yref="y",
            x=0.02,
            y=float(current_spot),
            text=f"Spot {float(current_spot):.2f}",
            showarrow=False,
            font=dict(color="#FFFFFF"),
            bgcolor="rgba(0,0,0,0.45)",
            xanchor="left",
            yanchor="top",
            xshift=4,
        )

    max_abs_x = max(curve["abs_weighted_gex"].max(), 1.0)

    for _, row in curve.iterrows():
        strike = float(row["strike"])
        gex_val = float(row["weighted_gex"])
        gamma_side = row["gamma_side"]

        if gamma_side == "SUPPORT":
            txt = "S"
            color = "#00E676"
        elif gamma_side == "RESISTANCE":
            txt = "R"
            color = "#FF5252"
        else:
            continue

        label_x = gex_val + (0.03 * max_abs_x if gex_val >= 0 else -0.03 * max_abs_x)
        fig.add_annotation(
            x=label_x,
            y=strike,
            text=txt,
            showarrow=False,
            font=dict(color=color, size=12),
            bgcolor="rgba(0,0,0,0.35)",
            xanchor="center",
            yanchor="middle",
        )

    shared_yaxis = get_shared_yaxis_config(forced_y_range)

    fig.update_layout(
        title=f"{ticker} - GEX by Strike (Hybrid)",
        xaxis_title="Weighted GEX",
        yaxis_title="Price / Strike",
        height=650,
        margin=dict(l=70, r=140, t=70, b=50),
        showlegend=False,
        yaxis=shared_yaxis,
    )

    fig.update_traces(cliponaxis=False)
    return fig, curve


def classify_consistent_trade_logic(oi_side, gamma_side, agreement, weighted_gex, weighted_vex):
    oi_side = str(oi_side or "OTHER")
    gamma_side = str(gamma_side or "OTHER")
    agreement = str(agreement or "OTHER")

    gex_val = 0.0 if pd.isna(weighted_gex) else float(weighted_gex)
    vex_val = 0.0 if pd.isna(weighted_vex) else float(weighted_vex)

    abs_vex = abs(vex_val)

    if abs_vex >= 25000:
        vex_strength = "HIGH"
    elif abs_vex >= 8000:
        vex_strength = "MEDIUM"
    elif abs_vex > 0:
        vex_strength = "LOW"
    else:
        vex_strength = "LOW"

    # -----------------------------
    # 1) Market behavior
    # -----------------------------
    if agreement == "ALIGNED":
        if vex_strength == "LOW":
            market_behavior = "CLEAN"
        elif vex_strength == "MEDIUM":
            market_behavior = "CONTROLLED"
        else:
            market_behavior = "FAST"
    elif agreement == "FLIP":
        if vex_strength == "HIGH":
            market_behavior = "EXPLOSIVE"
        elif vex_strength == "MEDIUM":
            market_behavior = "UNSTABLE"
        else:
            market_behavior = "CHOP"
    elif agreement == "GAMMA_ONLY":
        if vex_strength == "HIGH":
            market_behavior = "FAST"
        elif vex_strength == "MEDIUM":
            market_behavior = "CONTROLLED"
        else:
            market_behavior = "CLEAN"
    else:
        market_behavior = "CHOP"

    # -----------------------------
    # 2) Trade type + direction
    # -----------------------------
    best_trade_type = "SKIP"
    direction = "SKIP"

    # ALIGNED = bounce/scalp in direction of side
    if agreement == "ALIGNED":
        if gamma_side == "SUPPORT":
            direction = "LONG"
            if vex_strength == "LOW":
                best_trade_type = "BOUNCE"
            elif vex_strength == "MEDIUM":
                best_trade_type = "BOUNCE"
            else:
                best_trade_type = "SCALP"

        elif gamma_side == "RESISTANCE":
            direction = "SHORT"
            if vex_strength == "LOW":
                best_trade_type = "BOUNCE"
            elif vex_strength == "MEDIUM":
                best_trade_type = "BOUNCE"
            else:
                best_trade_type = "SCALP"

    # FLIP = breakout/breakdown if VEX is high enough; otherwise skip
    elif agreement == "FLIP":
        if vex_strength == "HIGH":
            if gamma_side == "SUPPORT":
                direction = "LONG"
                best_trade_type = "BREAKOUT"
            elif gamma_side == "RESISTANCE":
                direction = "SHORT"
                best_trade_type = "BREAKDOWN"
        elif vex_strength == "MEDIUM":
            if gamma_side == "SUPPORT":
                direction = "LONG"
                best_trade_type = "WATCH_BREAKOUT"
            elif gamma_side == "RESISTANCE":
                direction = "SHORT"
                best_trade_type = "WATCH_BREAKDOWN"
        else:
            direction = "SKIP"
            best_trade_type = "SKIP"

    # GAMMA_ONLY = follow gamma, but less confidence
    elif agreement == "GAMMA_ONLY":
        if gamma_side == "SUPPORT":
            direction = "LONG"
            best_trade_type = "SCALP" if vex_strength == "HIGH" else "BOUNCE"
        elif gamma_side == "RESISTANCE":
            direction = "SHORT"
            best_trade_type = "SCALP" if vex_strength == "HIGH" else "BOUNCE"

    # OTHER = skip
    else:
        direction = "SKIP"
        best_trade_type = "SKIP"

    # -----------------------------
    # 3) Final trade decision text
    # -----------------------------
    if best_trade_type == "SKIP" or direction == "SKIP":
        trade_decision = "SKIP"
    elif best_trade_type in ["WATCH_BREAKOUT", "WATCH_BREAKDOWN"]:
        trade_decision = best_trade_type
    else:
        trade_decision = f"{best_trade_type} {direction}"

    return {
        "vex_strength": vex_strength,
        "market_behavior": market_behavior,
        "best_trade_type": best_trade_type,
        "direction": direction,
        "trade_decision": trade_decision,
    }


def classify_consistent_trade_logic(oi_side, gamma_side, agreement, weighted_gex, weighted_vex):
    oi_side = str(oi_side or "OTHER")
    gamma_side = str(gamma_side or "OTHER")
    agreement = str(agreement or "OTHER")

    gex_val = 0.0 if pd.isna(weighted_gex) else float(weighted_gex)
    vex_val = 0.0 if pd.isna(weighted_vex) else float(weighted_vex)

    abs_vex = abs(vex_val)

    if abs_vex >= 25000:
        vex_strength = "HIGH"
    elif abs_vex >= 8000:
        vex_strength = "MEDIUM"
    elif abs_vex > 0:
        vex_strength = "LOW"
    else:
        vex_strength = "LOW"

    if agreement == "ALIGNED":
        if vex_strength == "LOW":
            market_behavior = "CLEAN"
        elif vex_strength == "MEDIUM":
            market_behavior = "CONTROLLED"
        else:
            market_behavior = "FAST"
    elif agreement == "FLIP":
        if vex_strength == "HIGH":
            market_behavior = "EXPLOSIVE"
        elif vex_strength == "MEDIUM":
            market_behavior = "UNSTABLE"
        else:
            market_behavior = "CHOP"
    elif agreement == "GAMMA_ONLY":
        if vex_strength == "HIGH":
            market_behavior = "FAST"
        elif vex_strength == "MEDIUM":
            market_behavior = "CONTROLLED"
        else:
            market_behavior = "CLEAN"
    else:
        market_behavior = "CHOP"

    best_trade_type = "SKIP"
    direction = "SKIP"

    if agreement == "ALIGNED":
        if gamma_side == "SUPPORT":
            direction = "LONG"
            if vex_strength == "HIGH":
                best_trade_type = "SCALP"
            else:
                best_trade_type = "BOUNCE"

        elif gamma_side == "RESISTANCE":
            direction = "SHORT"
            if vex_strength == "HIGH":
                best_trade_type = "SCALP"
            else:
                best_trade_type = "BOUNCE"

    elif agreement == "FLIP":
        if vex_strength == "HIGH":
            if gamma_side == "SUPPORT":
                direction = "LONG"
                best_trade_type = "BREAKOUT"
            elif gamma_side == "RESISTANCE":
                direction = "SHORT"
                best_trade_type = "BREAKDOWN"
        elif vex_strength == "MEDIUM":
            if gamma_side == "SUPPORT":
                direction = "LONG"
                best_trade_type = "WATCH_BREAKOUT"
            elif gamma_side == "RESISTANCE":
                direction = "SHORT"
                best_trade_type = "WATCH_BREAKDOWN"
        else:
            direction = "SKIP"
            best_trade_type = "SKIP"

    elif agreement == "GAMMA_ONLY":
        if gamma_side == "SUPPORT":
            direction = "LONG"
            best_trade_type = "SCALP" if vex_strength == "HIGH" else "BOUNCE"
        elif gamma_side == "RESISTANCE":
            direction = "SHORT"
            best_trade_type = "SCALP" if vex_strength == "HIGH" else "BOUNCE"

    else:
        direction = "SKIP"
        best_trade_type = "SKIP"

    if best_trade_type == "SKIP" or direction == "SKIP":
        trade_decision = "SKIP"
    elif best_trade_type in ["WATCH_BREAKOUT", "WATCH_BREAKDOWN"]:
        trade_decision = best_trade_type
    else:
        trade_decision = f"{best_trade_type} {direction}"

    return {
        "vex_strength": vex_strength,
        "market_behavior": market_behavior,
        "best_trade_type": best_trade_type,
        "direction": direction,
        "trade_decision": trade_decision,
    }


def enrich_gex_table(curve_df: pd.DataFrame, levels_df: pd.DataFrame, spot_price: float) -> pd.DataFrame:
    if curve_df.empty:
        return curve_df

    enriched_df = curve_df.copy()

    # Initialize columns
    enriched_df["oi_side"] = "OTHER"
    enriched_df["agreement"] = "OTHER"
    enriched_df["weighted_vex"] = None
    enriched_df["vex_strength"] = None
    enriched_df["market_behavior"] = None
    enriched_df["best_trade_type"] = None
    enriched_df["direction"] = None
    enriched_df["trade_decision"] = None
    enriched_df["Entry-Stop-Target"] = None
    enriched_df["distance_to_spot"] = None
    enriched_df["highlight_flag"] = ""

    # Prepare levels lookup
    if not levels_df.empty:
        levels_map = levels_df.copy()
        levels_map["level"] = pd.to_numeric(levels_map["level"], errors="coerce")
        levels_map = levels_map.dropna(subset=["level"])
        levels_map["level"] = levels_map["level"].astype(float)
        levels_map = levels_map.groupby("level").first()

        for idx, row in enriched_df.iterrows():
            strike = float(row["strike"])
            gamma_side = str(row.get("gamma_side", "OTHER"))

            oi_side = "OTHER"
            weighted_vex = 0.0

            if strike in levels_map.index:
                matched = levels_map.loc[strike]
                oi_side = matched.get("side", "OTHER")
                weighted_vex = matched.get("level_vex", 0.0)

            enriched_df.at[idx, "oi_side"] = oi_side
            enriched_df.at[idx, "weighted_vex"] = weighted_vex
            enriched_df.at[idx, "distance_to_spot"] = abs(strike - float(spot_price))

            # Agreement logic
            if oi_side == gamma_side and oi_side != "OTHER":
                agreement = "ALIGNED"
            elif oi_side == "OTHER" and gamma_side != "OTHER":
                agreement = "GAMMA_ONLY"
            elif oi_side != gamma_side and oi_side != "OTHER" and gamma_side != "OTHER":
                agreement = "FLIP"
            else:
                agreement = "OTHER"

            enriched_df.at[idx, "agreement"] = agreement

            # --- Trading logic ---
            vex_val = 0.0 if pd.isna(weighted_vex) else float(weighted_vex)
            abs_vex = abs(vex_val)

            # VEX strength
            if abs_vex >= 25000:
                vex_strength = "HIGH"
            elif abs_vex >= 8000:
                vex_strength = "MEDIUM"
            elif abs_vex > 0:
                vex_strength = "LOW"
            else:
                vex_strength = "LOW"

            # Market behavior
            if agreement == "ALIGNED":
                if vex_strength == "LOW":
                    market_behavior = "CLEAN"
                elif vex_strength == "MEDIUM":
                    market_behavior = "CONTROLLED"
                else:
                    market_behavior = "FAST"

            elif agreement == "FLIP":
                if vex_strength == "HIGH":
                    market_behavior = "EXPLOSIVE"
                elif vex_strength == "MEDIUM":
                    market_behavior = "UNSTABLE"
                else:
                    market_behavior = "CHOP"

            elif agreement == "GAMMA_ONLY":
                if vex_strength == "HIGH":
                    market_behavior = "FAST"
                elif vex_strength == "MEDIUM":
                    market_behavior = "CONTROLLED"
                else:
                    market_behavior = "CLEAN"
            else:
                market_behavior = "CHOP"

            # Trade decision
            best_trade_type = "SKIP"
            direction = "SKIP"

            if agreement == "ALIGNED":
                if gamma_side == "SUPPORT":
                    direction = "LONG"
                    best_trade_type = "SCALP" if vex_strength == "HIGH" else "BOUNCE"
                elif gamma_side == "RESISTANCE":
                    direction = "SHORT"
                    best_trade_type = "SCALP" if vex_strength == "HIGH" else "BOUNCE"

            elif agreement == "FLIP":
                if vex_strength == "HIGH":
                    if gamma_side == "SUPPORT":
                        direction = "LONG"
                        best_trade_type = "BREAKOUT"
                    elif gamma_side == "RESISTANCE":
                        direction = "SHORT"
                        best_trade_type = "BREAKDOWN"
                elif vex_strength == "MEDIUM":
                    if gamma_side == "SUPPORT":
                        direction = "LONG"
                        best_trade_type = "WATCH_BREAKOUT"
                    elif gamma_side == "RESISTANCE":
                        direction = "SHORT"
                        best_trade_type = "WATCH_BREAKDOWN"
                else:
                    direction = "SKIP"
                    best_trade_type = "SKIP"

            elif agreement == "GAMMA_ONLY":
                if gamma_side == "SUPPORT":
                    direction = "LONG"
                    best_trade_type = "SCALP" if vex_strength == "HIGH" else "BOUNCE"
                elif gamma_side == "RESISTANCE":
                    direction = "SHORT"
                    best_trade_type = "SCALP" if vex_strength == "HIGH" else "BOUNCE"

            # Final label
            if best_trade_type == "SKIP" or direction == "SKIP":
                trade_decision = "SKIP"
            elif best_trade_type in ["WATCH_BREAKOUT", "WATCH_BREAKDOWN"]:
                trade_decision = best_trade_type
            else:
                trade_decision = f"{best_trade_type} {direction}"

            # Entry / Stop / Target as one column
            est = "-"
            if direction != "SKIP" and best_trade_type not in ["SKIP", "WATCH_BREAKOUT", "WATCH_BREAKDOWN"]:
                if direction == "LONG":
                    if best_trade_type == "BOUNCE":
                        entry = strike + 0.10
                        stop = strike - 0.40
                        target = strike + 1.00
                    elif best_trade_type == "SCALP":
                        entry = strike + 0.05
                        stop = strike - 0.25
                        target = strike + 0.40
                    elif best_trade_type == "BREAKOUT":
                        entry = strike + 0.15
                        stop = strike - 0.35
                        target = strike + 1.20
                    else:
                        entry = stop = target = None
                else: # SHORT / BREAKDOWN
                    if best_trade_type == "BOUNCE":
                        entry = strike - 0.10
                        stop = strike + 0.40
                        target = strike - 1.00
                    elif best_trade_type == "SCALP":
                        entry = strike - 0.05
                        stop = strike + 0.25
                        target = strike - 0.40
                    elif best_trade_type == "BREAKDOWN":
                        entry = strike - 0.15
                        stop = strike + 0.35
                        target = strike - 1.20
                    else:
                        entry = stop = target = None

                if entry is not None:
                    est = f"{entry:.2f} - {stop:.2f} - {target:.2f}"

            enriched_df.at[idx, "vex_strength"] = vex_strength
            enriched_df.at[idx, "market_behavior"] = market_behavior
            enriched_df.at[idx, "best_trade_type"] = best_trade_type
            enriched_df.at[idx, "direction"] = direction
            enriched_df.at[idx, "trade_decision"] = trade_decision
            enriched_df.at[idx, "Entry-Stop-Target"] = est

    # Add SPOT row
    spot_row = {
        "strike": float(spot_price),
        "oi_side": "SPOT",
        "gamma_side": "SPOT",
        "agreement": "REFERENCE",
        "weighted_gex": 0,
        "weighted_vex": 0,
        "vex_strength": "-",
        "market_behavior": "-",
        "best_trade_type": "-",
        "direction": "-",
        "trade_decision": "WATCH",
        "Entry-Stop-Target": "-",
        "distance_to_spot": 0.0,
        "highlight_flag": "SPOT",
    }

    enriched_df = pd.concat(
        [enriched_df, pd.DataFrame([spot_row])],
        ignore_index=True
    )

    # Highlight nearest tradable levels
    tradable_mask = (
        (enriched_df["trade_decision"] != "SKIP") &
        (enriched_df["trade_decision"] != "WATCH") &
        (enriched_df["oi_side"] != "SPOT")
    )

    if tradable_mask.any():
        nearest_idx = (
            enriched_df.loc[tradable_mask]
            .sort_values(["distance_to_spot", "strike"], ascending=[True, False])
            .head(3)
            .index
        )
        enriched_df.loc[nearest_idx, "highlight_flag"] = "HOT"

    # Sort descending by strike
    enriched_df = enriched_df.sort_values("strike", ascending=False).reset_index(drop=True)

    # Remove helper column before showing if you want to keep the table cleaner later,
    # but leave it for styling logic if needed.
    return enriched_df


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

tab1, tab2, tab3 = st.tabs(["Dashboard", "Charts", "Hybrid View"])

ticker_data = {}

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
    st.write("Charts show the last 24 hours with different background colors for premarket, aftermarket, and overnight. Saturdays and Sundays are removed from the x-axis.")

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
                "vex_strength",
                "gamma_strength",
                "market_behavior",
                "best_trade_type",
                "direction",
                "static_score",
                "static_grade",
                "dynamic_score",
                "grade",
                "trade_now_signal",
                "bounce_probability",
                "breakout_probability",
                "entry",
                "stop",
                "target",
            ]
            cols = [c for c in cols if c in levels_df.columns]

            fig = build_chart_for_ticker(
                ticker=ticker,
                hist_df=hist,
                levels_df=levels_df,
                current_spot=float(data["gamma"]["spot"]),
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"{ticker}_main_price_chart",
            )

            st.write("### Level Summary")
            st.dataframe(levels_df[cols], use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"{ticker} chart error: {e}")

        st.divider()

def render_hybrid_gex_table(enriched_df: pd.DataFrame):
    if enriched_df.empty:
        st.warning("No strongest GEX table data available.")
        return

    def highlight_rows(row):
        if row.get("highlight_flag") == "SPOT":
            return ["background-color: rgba(100, 181, 246, 0.25)"] * len(row)
        if row.get("highlight_flag") == "HOT":
            return ["background-color: rgba(255, 215, 64, 0.25)"] * len(row)
        return [""] * len(row)

    display_cols = [
        "strike",
        "oi_side",
        "gamma_side",
        "agreement",
        "weighted_gex",
        "weighted_vex",
        "vex_strength",
        "market_behavior",
        "best_trade_type",
        "direction",
        "trade_decision",
        "Entry-Stop-Target",
    ]
    display_cols = [c for c in display_cols if c in enriched_df.columns]

    styled_df = (
        enriched_df[display_cols + ["highlight_flag"]]
        .head(20)
        .style
        .apply(highlight_rows, axis=1)
        .hide(axis="index")
        .hide(axis="columns", subset=["highlight_flag"])
    )

    st.dataframe(
        styled_df,
        use_container_width=True,
    )

with tab3:
    st.write("Hybrid view combines a 16-hour OI price chart with a GEX-by-strike chart inside one shared subplot figure, so levels line up exactly.")

    for ticker in (tickers or DEFAULT_TICKERS):
        st.header(f"{ticker} Hybrid View")

        data = ticker_data.get(ticker, {})
        if "error" in data:
            st.error(f"{ticker}: {data['error']}")
            st.divider()
            continue

        try:
            # -----------------------------
            # DATA PREP
            # -----------------------------
            hist_full = cached_intraday_history(ticker)
            hist_16h = slice_history_last_hours(hist_full, 16)

            levels_df = data["confluence"]["levels"].copy()
            gamma = data["gamma"]
            oi_key_level = data["oi_payload"].get("key_level")

            aligned_y_range = get_aligned_y_range(
                hist_df=hist_16h,
                levels_df=levels_df,
                current_spot=float(gamma["spot"]),
            )

            # -----------------------------
            # HYBRID CHART (SUBPLOT)
            # -----------------------------
            hybrid_fig, curve_df = build_hybrid_subplot_figure(
                ticker=ticker,
                hist_df=hist_16h,
                levels_df=levels_df,
                gamma=gamma,
                oi_key_level=oi_key_level,
                forced_y_range=aligned_y_range,
            )

            st.plotly_chart(
                hybrid_fig,
                use_container_width=True,
                key=f"{ticker}_hybrid_subplot_chart",
            )

            # -----------------------------
            # GEX TABLE (ENRICHED + HIGHLIGHTED)
            # -----------------------------
            st.write("### Strongest GEX Strikes")

            if not curve_df.empty:
                enriched_df = enrich_gex_table(
                    curve_df,
                    levels_df,
                    spot_price=float(gamma["spot"]),
                )

                render_hybrid_gex_table(enriched_df)

            else:
                st.warning(f"No strongest GEX table data available for {ticker}.")

            st.divider()

        except Exception as e:
            st.error(f"{ticker} hybrid view error: {e}")
            st.divider()
