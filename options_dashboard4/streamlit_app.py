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
def cached_gamma(ticker, weights, max_distance, num_levels, fixed_spot=None):
    return get_gamma_levels(
        ticker_symbol=ticker,
        weights=list(weights),
        max_distance=float(max_distance),
        num_levels=int(num_levels),
        fixed_spot=fixed_spot,
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
        "spy_x1": 0.0,
        "spy_x2": 0.0,
        "spy_y1": 0.0,
        "spy_y2": 0.0,
        "qqq_x1": 0.0,
        "qqq_x2": 0.0,
        "qqq_y1": 0.0,
        "qqq_y2": 0.0,
    }
    saved = load_json(SETTINGS_FILE, default_settings)
    return {
        "tickers": saved.get("tickers", DEFAULT_TICKERS),
        "weights": saved.get("weights", DEFAULT_EXPIRATION_WEIGHTS),
        "max_distance": saved.get("max_distance", DEFAULT_MAX_DISTANCE),
        "num_levels": saved.get("num_levels", DEFAULT_NUM_LEVELS),
        "spy_x1": float(saved.get("spy_x1", 0.0)),
        "spy_x2": float(saved.get("spy_x2", 0.0)),
        "spy_y1": float(saved.get("spy_y1", 0.0)),
        "spy_y2": float(saved.get("spy_y2", 0.0)),
        "qqq_x1": float(saved.get("qqq_x1", 0.0)),
        "qqq_x2": float(saved.get("qqq_x2", 0.0)),
        "qqq_y1": float(saved.get("qqq_y1", 0.0)),
        "qqq_y2": float(saved.get("qqq_y2", 0.0)),
    }

def calculate_regression_from_points(x1, x2, y1, y2):
    try:
        x1 = float(x1)
        x2 = float(x2)
        y1 = float(y1)
        y2 = float(y2)

        if x2 == x1:
            return None, None

        b = (y2 - y1) / (x2 - x1)
        a = y1 - (b * x1)
        return a, b
    except Exception:
        return None, None


def get_futures_equivalent_label(ticker):
    if ticker == "SPY":
        return "ES Equivalent"
    if ticker == "QQQ":
        return "MNQ Equivalent"
    return "Futures Equivalent"


def calculate_futures_equivalent(ticker, x_value, settings_dict):
    if pd.isna(x_value):
        return None

    if ticker == "SPY":
        a, b = calculate_regression_from_points(
            settings_dict.get("spy_x1", 0.0),
            settings_dict.get("spy_x2", 0.0),
            settings_dict.get("spy_y1", 0.0),
            settings_dict.get("spy_y2", 0.0),
        )
    elif ticker == "QQQ":
        a, b = calculate_regression_from_points(
            settings_dict.get("qqq_x1", 0.0),
            settings_dict.get("qqq_x2", 0.0),
            settings_dict.get("qqq_y1", 0.0),
            settings_dict.get("qqq_y2", 0.0),
        )
    else:
        return None

    if a is None or b is None:
        return None

    return round(a + b * float(x_value), 2)

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

    curve = pd.DataFrame(gamma.get("gex_curve", []))
    if curve.empty:
        curve = pd.DataFrame(gamma.get("gex_curve_wide", []))

    if curve.empty:
        return fig, pd.DataFrame()

    curve = curve.sort_values("strike").reset_index(drop=True)

    def classify_gamma_side(weighted_gex, strike, spot):
        if pd.isna(weighted_gex) or pd.isna(strike) or pd.isna(spot):
            return "OTHER"

        gex = float(weighted_gex)
        strike = float(strike)
        spot = float(spot)

        if gex > 0:
            if strike < spot:
                return "SUPPORT"
            elif strike > spot:
                return "RESISTANCE"
            return "SUPPORT"

        if gex < 0:
            if strike < spot:
                return "BREAKOUT-PRONE SUPPORT"
            elif strike > spot:
                return "BREAKOUT-PRONE RESISTANCE"
            return "BREAKOUT-PRONE"

        return "OTHER"

    curve["gamma_side"] = curve.apply(
        lambda row: classify_gamma_side(
            row["weighted_gex"],
            row["strike"],
            current_spot,
        ),
        axis=1,
    )
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

    gamma_key_local = gamma.get("gamma_key_local", gamma.get("key_level"))
    gamma_key_global = gamma.get("gamma_key_global")
    gamma_flip = gamma.get("gamma_flip")

    if gamma_key_local is not None:
        fig.add_hline(
            y=float(gamma_key_local),
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
            y=float(gamma_key_local),
            text=f"Gamma Key Local {float(gamma_key_local):.2f}",
            showarrow=False,
            font=dict(color="#FFD54F", size=11),
            bgcolor="rgba(0,0,0,0.35)",
            xanchor="right",
            yanchor="bottom",
        )

    if gamma_key_global is not None:
        fig.add_hline(
            y=float(gamma_key_global),
            line_width=2,
            line_dash="dot",
            line_color="#BA68C8",
            row=1,
            col=2,
        )
        fig.add_annotation(
            xref="paper",
            yref="y",
            x=0.985,
            y=float(gamma_key_global),
            text=f"Gamma Key Global {float(gamma_key_global):.2f}",
            showarrow=False,
            font=dict(color="#BA68C8", size=11),
            bgcolor="rgba(0,0,0,0.35)",
            xanchor="right",
            yanchor="top",
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
            yanchor="middle",
        )

    max_abs_x = max(curve["abs_weighted_gex"].max(), 1.0)

    for _, row in curve.iterrows():
        strike = float(row["strike"])
        gex_val = float(row["weighted_gex"])
        gamma_side = str(row["gamma_side"])

        if "SUPPORT" in gamma_side:
            txt = "S"
            color = "#00E676"
        elif "RESISTANCE" in gamma_side:
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
    current_spot = float(gamma["spot"])

    def classify_gamma_side(weighted_gex, strike, spot):
        if pd.isna(weighted_gex) or pd.isna(strike) or pd.isna(spot):
            return "OTHER"

        gex = float(weighted_gex)
        strike = float(strike)
        spot = float(spot)

        if gex > 0:
            if strike < spot:
                return "SUPPORT"
            elif strike > spot:
                return "RESISTANCE"
            return "SUPPORT"

        if gex < 0:
            if strike < spot:
                return "BREAKOUT-PRONE SUPPORT"
            elif strike > spot:
                return "BREAKOUT-PRONE RESISTANCE"
            return "BREAKOUT-PRONE"

        return "OTHER"

    curve["gamma_side"] = curve.apply(
        lambda row: classify_gamma_side(
            row["weighted_gex"],
            row["strike"],
            current_spot,
        ),
        axis=1,
    )
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

    gamma_key_local = gamma.get("gamma_key_local", gamma.get("key_level"))
    gamma_key_global = gamma.get("gamma_key_global")
    gamma_flip = gamma.get("gamma_flip")
    current_spot = gamma.get("spot")

    if gamma_key_local is not None:
        fig.add_hline(
            y=float(gamma_key_local),
            line_width=2,
            line_dash="dash",
            line_color="#FFD54F",
        )
        fig.add_annotation(
            xref="paper",
            yref="y",
            x=0.98,
            y=float(gamma_key_local),
            text=f"Gamma Key Local {float(gamma_key_local):.2f}",
            showarrow=False,
            font=dict(color="#FFD54F"),
            bgcolor="rgba(0,0,0,0.35)",
            xanchor="right",
            yanchor="bottom",
            xshift=-4,
        )

    if gamma_key_global is not None:
        fig.add_hline(
            y=float(gamma_key_global),
            line_width=2,
            line_dash="dot",
            line_color="#BA68C8",
        )
        fig.add_annotation(
            xref="paper",
            yref="y",
            x=0.98,
            y=float(gamma_key_global),
            text=f"Gamma Key Global {float(gamma_key_global):.2f}",
            showarrow=False,
            font=dict(color="#BA68C8"),
            bgcolor="rgba(0,0,0,0.35)",
            xanchor="right",
            yanchor="top",
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
            yanchor="middle",
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
        gamma_side = str(row["gamma_side"])

        if "SUPPORT" in gamma_side:
            txt = "S"
            color = "#00E676"
        elif "RESISTANCE" in gamma_side:
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


def enrich_gex_table(
    curve_df: pd.DataFrame,
    levels_df: pd.DataFrame,
    spot_price: float,
    ticker: str,
    settings_dict: dict,
    gamma: dict,
    oi_key_level,
) -> pd.DataFrame:
    if curve_df.empty:
        return curve_df

    enriched_df = curve_df.copy()

    enriched_df["oi_side"] = "OTHER"
    enriched_df["agreement"] = "OTHER"
    enriched_df["weighted_vex"] = 0.0
    enriched_df["vex_strength"] = "LOW"
    enriched_df["market_behavior"] = "CHOP"
    enriched_df["best_trade_type"] = "SKIP"
    enriched_df["direction"] = "SKIP"
    enriched_df["trade_decision"] = "SKIP"
    enriched_df["Entry-Stop-Target"] = "-"
    enriched_df["distance_to_spot"] = 0.0
    enriched_df["highlight_flag"] = ""
    enriched_df["Futures Equivalent"] = None
    enriched_df["auto_flag"] = "SKIP"
    enriched_df["decision_reason"] = ""
    enriched_df["key_interaction"] = "NONE"
    enriched_df["trade_score"] = 0.0
    enriched_df["gamma_regime"] = "UNKNOWN"
    enriched_df["trigger_state"] = "NONE"
    enriched_df["breakout_risk"] = "LOW"

    gamma_key = gamma.get("key_level")
    gamma_flip = gamma.get("gamma_flip")
    gamma_regime_payload = str(gamma.get("regime", "UNKNOWN"))

    if not levels_df.empty:
        levels_map = levels_df.copy()
        levels_map["level"] = pd.to_numeric(levels_map["level"], errors="coerce")
        levels_map = levels_map.dropna(subset=["level"])
        levels_map["level"] = levels_map["level"].astype(float)
        levels_map = levels_map.groupby("level").first()
    else:
        levels_map = pd.DataFrame()

    max_abs_gex = 0.0
    if "weighted_gex" in enriched_df.columns:
        max_abs_gex = float(pd.to_numeric(enriched_df["weighted_gex"], errors="coerce").abs().max())
        if pd.isna(max_abs_gex):
            max_abs_gex = 0.0

    def classify_gamma_side(weighted_gex, strike, spot):
        if pd.isna(weighted_gex) or pd.isna(strike) or pd.isna(spot):
            return "OTHER"

        gex = float(weighted_gex)
        strike = float(strike)
        spot = float(spot)

        if gex > 0:
            if strike < spot:
                return "SUPPORT"
            elif strike > spot:
                return "RESISTANCE"
            return "SUPPORT"

        if gex < 0:
            if strike < spot:
                return "BREAKOUT-PRONE SUPPORT"
            elif strike > spot:
                return "BREAKOUT-PRONE RESISTANCE"
            return "BREAKOUT-PRONE"

        return "OTHER"

    def gamma_direction(gamma_side):
        gamma_side = str(gamma_side or "OTHER")
        if "SUPPORT" in gamma_side:
            return "SUPPORT"
        if "RESISTANCE" in gamma_side:
            return "RESISTANCE"
        return "OTHER"

    def gamma_is_breakout_prone(gamma_side):
        return "BREAKOUT-PRONE" in str(gamma_side or "")

    def get_threshold():
        return 0.5 if ticker == "SPY" else 1.0

    def get_confirm_buffer():
        return 0.20 if ticker == "SPY" else 0.50

    def classify_vex_strength(v):
        av = abs(float(v))
        if av >= 25000:
            return "HIGH"
        if av >= 8000:
            return "MEDIUM"
        return "LOW"

    def near_level(x, lvl, threshold):
        if lvl is None or pd.isna(lvl):
            return False
        return abs(float(x) - float(lvl)) <= float(threshold)

    def get_key_interaction(strike, spot):
        threshold = get_threshold()
        interactions = []

        if near_level(strike, gamma_key, threshold):
            interactions.append("AT_GAMMA_KEY")
        if near_level(strike, gamma_flip, threshold):
            interactions.append("AT_GAMMA_FLIP")
        if near_level(strike, oi_key_level, threshold):
            interactions.append("AT_OI_KEY")
        if near_level(spot, gamma_key, threshold):
            interactions.append("SPOT_NEAR_GAMMA_KEY")
        if near_level(spot, gamma_flip, threshold):
            interactions.append("SPOT_NEAR_GAMMA_FLIP")
        if near_level(spot, oi_key_level, threshold):
            interactions.append("SPOT_NEAR_OI_KEY")

        return " | ".join(interactions) if interactions else "NONE"

    def get_regime_side(spot):
        if gamma_flip is not None and not pd.isna(gamma_flip):
            if float(spot) < float(gamma_flip):
                return "SHORT_GAMMA"
            if float(spot) > float(gamma_flip):
                return "LONG_GAMMA"
            return "AT_FLIP"

        if gamma_regime_payload == "NO_LOCAL_FLIP_LONG_GAMMA_BIAS":
            return "LONG_GAMMA"
        if gamma_regime_payload == "NO_LOCAL_FLIP_SHORT_GAMMA_BIAS":
            return "SHORT_GAMMA"
        return "UNKNOWN"

    def score_setup(
        agreement,
        vex_strength,
        key_interaction,
        weighted_gex,
        strike,
        spot,
        auto_flag,
        gamma_regime,
        trigger_state,
        breakout_risk,
    ):
        score = 0.0

        if agreement == "ALIGNED":
            score += 30
        elif agreement == "GAMMA_ONLY":
            score += 20
        elif agreement == "FLIP":
            if vex_strength == "HIGH":
                score += 18
            elif vex_strength == "MEDIUM":
                score += 12
            else:
                score += 4

        gex_abs = abs(float(weighted_gex)) if not pd.isna(weighted_gex) else 0.0
        if max_abs_gex > 0:
            score += min((gex_abs / max_abs_gex) * 25.0, 25.0)

        dist = abs(float(strike) - float(spot))
        if ticker == "SPY":
            if dist <= 0.5:
                score += 15
            elif dist <= 1.0:
                score += 12
            elif dist <= 2.0:
                score += 8
            elif dist <= 4.0:
                score += 4
        else:
            if dist <= 1.0:
                score += 15
            elif dist <= 2.0:
                score += 12
            elif dist <= 4.0:
                score += 8
            elif dist <= 8.0:
                score += 4

        expansion_flags = {
            "BREAKOUT_READY", "BREAKDOWN_READY",
            "TRAP_UP", "TRAP_DOWN",
            "PRE_BREAKOUT", "PRE_BREAKDOWN",
            "WATCH_UP", "WATCH_DOWN",
        }
        if auto_flag in expansion_flags:
            if vex_strength == "HIGH":
                score += 10
            elif vex_strength == "MEDIUM":
                score += 6
            else:
                score += 1
        else:
            if vex_strength == "LOW":
                score += 10
            elif vex_strength == "MEDIUM":
                score += 7
            else:
                score += 4

        if "AT_GAMMA_KEY" in key_interaction and "AT_OI_KEY" in key_interaction:
            score += 10
        elif "AT_GAMMA_FLIP" in key_interaction or "SPOT_NEAR_GAMMA_FLIP" in key_interaction:
            score += 8
        elif "AT_OI_KEY" in key_interaction or "AT_GAMMA_KEY" in key_interaction:
            score += 6
        elif "SPOT_NEAR_GAMMA_KEY" in key_interaction:
            score += 4

        bounce_flags = {
            "A_PLUS_SUPPORT", "A_PLUS_RESISTANCE",
            "BOUNCE_LONG", "BOUNCE_SHORT",
            "FAST_LONG", "FAST_SHORT",
            "SCALP_ONLY",
            "GAMMA_ONLY_LONG", "GAMMA_ONLY_SHORT",
        }

        if gamma_regime == "SHORT_GAMMA":
            if auto_flag in expansion_flags:
                score += 10
            elif auto_flag in bounce_flags:
                score -= 4
        elif gamma_regime == "LONG_GAMMA":
            if auto_flag in expansion_flags:
                score -= 8
            elif auto_flag in bounce_flags:
                score += 8
        elif gamma_regime == "AT_FLIP":
            score -= 3

        if trigger_state in ["CONFIRMED_BREAKOUT", "CONFIRMED_BREAKDOWN"]:
            score += 6
        elif trigger_state in ["PRE_BREAKOUT", "PRE_BREAKDOWN"]:
            score += 3
        elif trigger_state == "REJECTION_ZONE":
            score += 2

        if breakout_risk == "HIGH":
            score += 4
        elif breakout_risk == "MEDIUM":
            score += 2

        return round(min(max(score, 0.0), 100.0), 1)

    def classify_core_logic(oi_side, gamma_side, weighted_gex, weighted_vex, strike, spot):
        vex_strength = classify_vex_strength(weighted_vex)
        gamma_regime = get_regime_side(spot)
        threshold = get_threshold()
        confirm_buffer = get_confirm_buffer()

        gamma_dir = gamma_direction(gamma_side)
        breakout_prone = gamma_is_breakout_prone(gamma_side)

        if oi_side == gamma_dir and oi_side != "OTHER":
            agreement = "ALIGNED"
        elif oi_side == "OTHER" and gamma_dir != "OTHER":
            agreement = "GAMMA_ONLY"
        elif oi_side != gamma_dir and oi_side != "OTHER" and gamma_dir != "OTHER":
            agreement = "FLIP"
        else:
            agreement = "OTHER"

        key_interaction = get_key_interaction(strike, spot)

        if gamma_regime == "SHORT_GAMMA":
            if vex_strength == "HIGH":
                market_behavior = "EXPANSIVE"
            elif vex_strength == "MEDIUM":
                market_behavior = "TRENDING"
            else:
                market_behavior = "UNSTABLE"
        elif gamma_regime == "LONG_GAMMA":
            if vex_strength == "LOW":
                market_behavior = "CLEAN"
            elif vex_strength == "MEDIUM":
                market_behavior = "CONTROLLED"
            else:
                market_behavior = "FAST"
        else:
            market_behavior = "TRANSITION"

        best_trade_type = "SKIP"
        direction = "SKIP"
        auto_flag = "SKIP"
        decision_reason = "Weak or conflicting setup."
        trigger_state = "NONE"
        breakout_risk = "LOW"

        if agreement == "ALIGNED" and gamma_regime == "SHORT_GAMMA" and not breakout_prone:
            if vex_strength == "HIGH" and abs(float(strike) - float(spot)) <= threshold * 2:
                breakout_risk = "HIGH"
            elif vex_strength == "MEDIUM" and abs(float(strike) - float(spot)) <= threshold * 2:
                breakout_risk = "MEDIUM"

        if agreement == "FLIP" and oi_side == "RESISTANCE" and gamma_dir == "SUPPORT":
            if vex_strength in ["MEDIUM", "HIGH"]:
                if float(spot) < float(strike) and abs(float(strike) - float(spot)) <= threshold * 2:
                    trigger_state = "PRE_BREAKOUT"
                elif float(spot) >= float(strike) + confirm_buffer:
                    trigger_state = "CONFIRMED_BREAKOUT"

        elif agreement == "FLIP" and oi_side == "SUPPORT" and gamma_dir == "RESISTANCE":
            if vex_strength in ["MEDIUM", "HIGH"]:
                if float(spot) > float(strike) and abs(float(strike) - float(spot)) <= threshold * 2:
                    trigger_state = "PRE_BREAKDOWN"
                elif float(spot) <= float(strike) - confirm_buffer:
                    trigger_state = "CONFIRMED_BREAKDOWN"

        elif agreement == "ALIGNED" and not breakout_prone:
            if gamma_dir == "RESISTANCE" and float(spot) <= float(strike):
                trigger_state = "REJECTION_ZONE"
            elif gamma_dir == "SUPPORT" and float(spot) >= float(strike):
                trigger_state = "REJECTION_ZONE"

        if "SPOT_NEAR_GAMMA_KEY" in key_interaction and agreement in ["ALIGNED", "GAMMA_ONLY"] and not breakout_prone:
            market_behavior = "PINNED"
            if gamma_dir == "SUPPORT":
                direction = "LONG"
                best_trade_type = "SCALP"
                auto_flag = "SCALP_ONLY"
                decision_reason = "Spot near Gamma Key: pinning/magnet behavior, prefer quick long scalps."
            elif gamma_dir == "RESISTANCE":
                direction = "SHORT"
                best_trade_type = "SCALP"
                auto_flag = "SCALP_ONLY"
                decision_reason = "Spot near Gamma Key: pinning/magnet behavior, prefer quick short scalps."

        elif "SPOT_NEAR_GAMMA_FLIP" in key_interaction:
            if vex_strength == "HIGH":
                if gamma_dir == "SUPPORT":
                    direction = "LONG"
                    best_trade_type = "WATCH_BREAKOUT"
                    auto_flag = "WATCH_UP"
                    decision_reason = "Spot near Gamma Flip with high VEX and gamma support: watch for upside expansion."
                elif gamma_dir == "RESISTANCE":
                    direction = "SHORT"
                    best_trade_type = "WATCH_BREAKDOWN"
                    auto_flag = "WATCH_DOWN"
                    decision_reason = "Spot near Gamma Flip with high VEX and gamma resistance: watch for downside expansion."
                else:
                    auto_flag = "TRANSITION_SKIP"
                    decision_reason = "Near Gamma Flip with no directional gamma side: transition zone."
            else:
                auto_flag = "TRANSITION_SKIP"
                decision_reason = "Near Gamma Flip without enough VEX: transition/chop risk."

        elif "AT_GAMMA_KEY" in key_interaction and "AT_OI_KEY" in key_interaction and agreement == "ALIGNED" and not breakout_prone:
            if gamma_dir == "SUPPORT":
                direction = "LONG"
                best_trade_type = "SCALP" if vex_strength == "HIGH" else "BOUNCE"
                auto_flag = "A_PLUS_SUPPORT"
                decision_reason = "Gamma Key and OI Key aligned at support: strongest long reaction zone."
            elif gamma_dir == "RESISTANCE":
                direction = "SHORT"
                best_trade_type = "SCALP" if vex_strength == "HIGH" else "BOUNCE"
                auto_flag = "A_PLUS_RESISTANCE"
                decision_reason = "Gamma Key and OI Key aligned at resistance: strongest short reaction zone."

        elif trigger_state == "CONFIRMED_BREAKOUT":
            direction = "LONG"
            best_trade_type = "BREAKOUT"
            auto_flag = "TRAP_UP"
            decision_reason = "Conflict level accepted above resistance: breakout/squeeze is confirmed."

        elif trigger_state == "PRE_BREAKOUT":
            direction = "LONG"
            best_trade_type = "WATCH_BREAKOUT"
            auto_flag = "PRE_BREAKOUT"
            decision_reason = "OI says resistance but gamma supports it. Price is still below the level: breakout may be building, not confirmed yet."

        elif trigger_state == "CONFIRMED_BREAKDOWN":
            direction = "SHORT"
            best_trade_type = "BREAKDOWN"
            auto_flag = "TRAP_DOWN"
            decision_reason = "Conflict level accepted below support: breakdown is confirmed."

        elif trigger_state == "PRE_BREAKDOWN":
            direction = "SHORT"
            best_trade_type = "WATCH_BREAKDOWN"
            auto_flag = "PRE_BREAKDOWN"
            decision_reason = "OI says support but gamma resists it. Price is still above the level: breakdown may be building, not confirmed yet."

        elif agreement == "FLIP":
            if vex_strength == "HIGH":
                if gamma_dir == "SUPPORT":
                    direction = "LONG"
                    best_trade_type = "WATCH_BREAKOUT"
                    auto_flag = "WATCH_UP"
                    decision_reason = "OI and GEX conflict with high VEX: upside squeeze possible, wait for confirmation."
                elif gamma_dir == "RESISTANCE":
                    direction = "SHORT"
                    best_trade_type = "WATCH_BREAKDOWN"
                    auto_flag = "WATCH_DOWN"
                    decision_reason = "OI and GEX conflict with high VEX: downside break possible, wait for confirmation."
            elif vex_strength == "MEDIUM":
                if gamma_dir == "SUPPORT":
                    direction = "LONG"
                    best_trade_type = "WATCH_BREAKOUT"
                    auto_flag = "WATCH_UP"
                    decision_reason = "Conflict with medium VEX: watch upside resolution."
                elif gamma_dir == "RESISTANCE":
                    direction = "SHORT"
                    best_trade_type = "WATCH_BREAKDOWN"
                    auto_flag = "WATCH_DOWN"
                    decision_reason = "Conflict with medium VEX: watch downside resolution."
            else:
                auto_flag = "CHOP_SKIP"
                decision_reason = "OI/GEX conflict with low VEX: chop/trap conditions."

        elif agreement == "ALIGNED":
            if breakout_prone:
                if gamma_dir == "SUPPORT":
                    direction = "SHORT"
                    if vex_strength in ["MEDIUM", "HIGH"]:
                        best_trade_type = "WATCH_BREAKDOWN"
                        auto_flag = "WATCH_DOWN"
                        decision_reason = "Breakout-prone support can fail. Watch for downside break."
                    else:
                        best_trade_type = "SKIP"
                        auto_flag = "CHOP_SKIP"
                        decision_reason = "Breakout-prone support with low VEX: messy conditions."
                elif gamma_dir == "RESISTANCE":
                    direction = "LONG"
                    if vex_strength in ["MEDIUM", "HIGH"]:
                        best_trade_type = "WATCH_BREAKOUT"
                        auto_flag = "WATCH_UP"
                        decision_reason = "Breakout-prone resistance can fail. Watch for upside break."
                    else:
                        best_trade_type = "SKIP"
                        auto_flag = "CHOP_SKIP"
                        decision_reason = "Breakout-prone resistance with low VEX: messy conditions."
            else:
                if gamma_dir == "SUPPORT":
                    direction = "LONG"
                    if vex_strength == "HIGH":
                        best_trade_type = "SCALP"
                        auto_flag = "FAST_LONG"
                        if breakout_risk == "HIGH":
                            decision_reason = "Aligned support with high VEX in short gamma: long scalp is valid, but support-failure / breakdown risk is elevated."
                        else:
                            decision_reason = "Aligned support with high VEX: fast long scalp."
                    else:
                        best_trade_type = "BOUNCE"
                        auto_flag = "BOUNCE_LONG"
                        if breakout_risk == "HIGH":
                            decision_reason = "Aligned support, but short-gamma environment raises breakdown risk. Respect bounce, but stay alert."
                        else:
                            decision_reason = "Aligned support: clean bounce-long setup."

                elif gamma_dir == "RESISTANCE":
                    direction = "SHORT"
                    if vex_strength == "HIGH":
                        best_trade_type = "SCALP"
                        auto_flag = "FAST_SHORT"
                        if breakout_risk == "HIGH":
                            decision_reason = "Aligned resistance with high VEX in short gamma: short scalp is valid, but breakout-failure risk is elevated."
                        else:
                            decision_reason = "Aligned resistance with high VEX: fast short scalp."
                    else:
                        best_trade_type = "BOUNCE"
                        auto_flag = "BOUNCE_SHORT"
                        if breakout_risk == "HIGH":
                            decision_reason = "Aligned resistance, but short-gamma environment raises breakout risk. Respect rejection, but stay alert."
                        else:
                            decision_reason = "Aligned resistance: clean bounce-short setup."

        elif agreement == "GAMMA_ONLY":
            if breakout_prone:
                if gamma_dir == "SUPPORT":
                    direction = "SHORT"
                    if vex_strength in ["MEDIUM", "HIGH"]:
                        best_trade_type = "WATCH_BREAKDOWN"
                        auto_flag = "WATCH_DOWN"
                        decision_reason = "Gamma-only breakout-prone support: watch for support failure."
                    else:
                        best_trade_type = "SKIP"
                        auto_flag = "CHOP_SKIP"
                        decision_reason = "Gamma-only breakout-prone support with low VEX: skip."
                elif gamma_dir == "RESISTANCE":
                    direction = "LONG"
                    if vex_strength in ["MEDIUM", "HIGH"]:
                        best_trade_type = "WATCH_BREAKOUT"
                        auto_flag = "WATCH_UP"
                        decision_reason = "Gamma-only breakout-prone resistance: watch for breakout."
                    else:
                        best_trade_type = "SKIP"
                        auto_flag = "CHOP_SKIP"
                        decision_reason = "Gamma-only breakout-prone resistance with low VEX: skip."
            else:
                if gamma_dir == "SUPPORT":
                    direction = "LONG"
                    best_trade_type = "SCALP" if vex_strength == "HIGH" else "BOUNCE"
                    auto_flag = "GAMMA_ONLY_LONG"
                    decision_reason = "Gamma-only support: lower-confidence long setup."
                elif gamma_dir == "RESISTANCE":
                    direction = "SHORT"
                    best_trade_type = "SCALP" if vex_strength == "HIGH" else "BOUNCE"
                    auto_flag = "GAMMA_ONLY_SHORT"
                    decision_reason = "Gamma-only resistance: lower-confidence short setup."

        if best_trade_type == "SKIP" or direction == "SKIP":
            trade_decision = "SKIP"
        elif best_trade_type in ["WATCH_BREAKOUT", "WATCH_BREAKDOWN"]:
            trade_decision = best_trade_type
        else:
            trade_decision = f"{best_trade_type} {direction}"

        trade_score = score_setup(
            agreement=agreement,
            vex_strength=vex_strength,
            key_interaction=key_interaction,
            weighted_gex=weighted_gex,
            strike=strike,
            spot=spot,
            auto_flag=auto_flag,
            gamma_regime=gamma_regime,
            trigger_state=trigger_state,
            breakout_risk=breakout_risk,
        )

        return {
            "agreement": agreement,
            "vex_strength": vex_strength,
            "market_behavior": market_behavior,
            "best_trade_type": best_trade_type,
            "direction": direction,
            "trade_decision": trade_decision,
            "auto_flag": auto_flag,
            "decision_reason": decision_reason,
            "key_interaction": key_interaction,
            "trade_score": trade_score,
            "gamma_regime": gamma_regime,
            "trigger_state": trigger_state,
            "breakout_risk": breakout_risk,
        }

    def build_est(direction, trade_type, strike, spot, behavior, auto_flag, gamma_regime):
        if direction == "SKIP" or trade_type in ["SKIP", "WATCH_BREAKOUT", "WATCH_BREAKDOWN"]:
            return "-"

        if ticker == "SPY":
            base_entry = 0.10
            tight_stop = 0.35
            med_stop = 0.55
            wide_stop = 0.85
            bounce_target = 0.90
            scalp_target = 0.45
            breakout_target = 1.40
        else:
            base_entry = 0.20
            tight_stop = 1.00
            med_stop = 1.60
            wide_stop = 2.50
            bounce_target = 2.50
            scalp_target = 1.20
            breakout_target = 4.00

        if auto_flag == "SCALP_ONLY" or behavior in ["PINNED", "FAST"]:
            stop_pad = tight_stop
            target_pad = scalp_target
            entry_pad = base_entry * 0.5
        elif auto_flag in ["TRAP_UP", "TRAP_DOWN"] or trade_type in ["BREAKOUT", "BREAKDOWN"]:
            stop_pad = med_stop if gamma_regime == "LONG_GAMMA" else wide_stop
            target_pad = breakout_target
            entry_pad = base_entry * 1.25
        elif auto_flag in ["A_PLUS_SUPPORT", "A_PLUS_RESISTANCE", "BOUNCE_LONG", "BOUNCE_SHORT"]:
            stop_pad = tight_stop
            target_pad = bounce_target
            entry_pad = base_entry
        else:
            stop_pad = med_stop
            target_pad = bounce_target
            entry_pad = base_entry

        if direction == "LONG":
            entry = strike + entry_pad
            stop = strike - stop_pad
            target = strike + target_pad
        else:
            entry = strike - entry_pad
            stop = strike + stop_pad
            target = strike - target_pad

        return f"{entry:.2f} - {stop:.2f} - {target:.2f}"

    for idx, row in enriched_df.iterrows():
        strike = float(row["strike"])
        weighted_gex = row.get("weighted_gex", 0.0)

        gamma_side = classify_gamma_side(
            weighted_gex=weighted_gex,
            strike=strike,
            spot=float(spot_price),
        )

        oi_side = "OTHER"
        weighted_vex = 0.0

        if not levels_map.empty and strike in levels_map.index:
            matched = levels_map.loc[strike]
            oi_side = matched.get("side", "OTHER")
            weighted_vex = matched.get("level_vex", 0.0)

        logic = classify_core_logic(
            oi_side=oi_side,
            gamma_side=gamma_side,
            weighted_gex=weighted_gex,
            weighted_vex=weighted_vex,
            strike=strike,
            spot=float(spot_price),
        )

        enriched_df.at[idx, "gamma_side"] = gamma_side
        enriched_df.at[idx, "oi_side"] = oi_side
        enriched_df.at[idx, "weighted_vex"] = weighted_vex
        enriched_df.at[idx, "distance_to_spot"] = abs(strike - float(spot_price))
        enriched_df.at[idx, "agreement"] = logic["agreement"]
        enriched_df.at[idx, "vex_strength"] = logic["vex_strength"]
        enriched_df.at[idx, "market_behavior"] = logic["market_behavior"]
        enriched_df.at[idx, "best_trade_type"] = logic["best_trade_type"]
        enriched_df.at[idx, "direction"] = logic["direction"]
        enriched_df.at[idx, "trade_decision"] = logic["trade_decision"]
        enriched_df.at[idx, "auto_flag"] = logic["auto_flag"]
        enriched_df.at[idx, "decision_reason"] = logic["decision_reason"]
        enriched_df.at[idx, "key_interaction"] = logic["key_interaction"]
        enriched_df.at[idx, "trade_score"] = logic["trade_score"]
        enriched_df.at[idx, "gamma_regime"] = logic["gamma_regime"]
        enriched_df.at[idx, "trigger_state"] = logic["trigger_state"]
        enriched_df.at[idx, "breakout_risk"] = logic["breakout_risk"]

        enriched_df.at[idx, "Entry-Stop-Target"] = build_est(
            direction=logic["direction"],
            trade_type=logic["best_trade_type"],
            strike=strike,
            spot=float(spot_price),
            behavior=logic["market_behavior"],
            auto_flag=logic["auto_flag"],
            gamma_regime=logic["gamma_regime"],
        )

        enriched_df.at[idx, "Futures Equivalent"] = calculate_futures_equivalent(
            ticker=ticker,
            x_value=strike,
            settings_dict=settings_dict,
        )

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
        "Futures Equivalent": calculate_futures_equivalent(
            ticker=ticker,
            x_value=float(spot_price),
            settings_dict=settings_dict,
        ),
        "auto_flag": "SPOT",
        "decision_reason": "Current spot reference.",
        "key_interaction": get_key_interaction(float(spot_price), float(spot_price)),
        "trade_score": 0.0,
        "gamma_regime": get_regime_side(float(spot_price)),
        "trigger_state": "NONE",
        "breakout_risk": "LOW",
    }

    enriched_df = pd.concat([enriched_df, pd.DataFrame([spot_row])], ignore_index=True)

    tradable_mask = (
        (enriched_df["trade_decision"] != "SKIP") &
        (enriched_df["trade_decision"] != "WATCH") &
        (enriched_df["oi_side"] != "SPOT")
    )

    if tradable_mask.any():
        nearest_idx = (
            enriched_df.loc[tradable_mask]
            .sort_values(["trade_score", "distance_to_spot", "strike"], ascending=[False, True, False])
            .head(3)
            .index
        )
        enriched_df.loc[nearest_idx, "highlight_flag"] = "HOT"

    enriched_df = enriched_df.sort_values("strike", ascending=False).reset_index(drop=True)
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

st.sidebar.markdown("---")
st.sidebar.subheader("SPY → ES Regression Inputs")

spy_x1 = st.sidebar.number_input("SPY X1", value=float(settings["spy_x1"]), step=0.01, format="%.6f")
spy_x2 = st.sidebar.number_input("SPY X2", value=float(settings["spy_x2"]), step=0.01, format="%.6f")
spy_y1 = st.sidebar.number_input("ES Y1", value=float(settings["spy_y1"]), step=0.01, format="%.6f")
spy_y2 = st.sidebar.number_input("ES Y2", value=float(settings["spy_y2"]), step=0.01, format="%.6f")

spy_a, spy_b = calculate_regression_from_points(spy_x1, spy_x2, spy_y1, spy_y2)
if spy_a is not None and spy_b is not None:
    st.sidebar.caption(f"SPY→ES: a = {spy_a:.6f}, b = {spy_b:.6f}")
else:
    st.sidebar.caption("SPY→ES: invalid points (X1 and X2 cannot be equal)")

st.sidebar.markdown("---")
st.sidebar.subheader("QQQ → MNQ Regression Inputs")

qqq_x1 = st.sidebar.number_input("QQQ X1", value=float(settings["qqq_x1"]), step=0.01, format="%.6f")
qqq_x2 = st.sidebar.number_input("QQQ X2", value=float(settings["qqq_x2"]), step=0.01, format="%.6f")
qqq_y1 = st.sidebar.number_input("MNQ Y1", value=float(settings["qqq_y1"]), step=0.01, format="%.6f")
qqq_y2 = st.sidebar.number_input("MNQ Y2", value=float(settings["qqq_y2"]), step=0.01, format="%.6f")

qqq_a, qqq_b = calculate_regression_from_points(qqq_x1, qqq_x2, qqq_y1, qqq_y2)
if qqq_a is not None and qqq_b is not None:
    st.sidebar.caption(f"QQQ→MNQ: a = {qqq_a:.6f}, b = {qqq_b:.6f}")
else:
    st.sidebar.caption("QQQ→MNQ: invalid points (X1 and X2 cannot be equal)")

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
        "spy_x1": float(spy_x1),
        "spy_x2": float(spy_x2),
        "spy_y1": float(spy_y1),
        "spy_y2": float(spy_y2),
        "qqq_x1": float(qqq_x1),
        "qqq_x2": float(qqq_x2),
        "qqq_y1": float(qqq_y1),
        "qqq_y2": float(qqq_y2),
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
            oi_payload.get("oi_fixed_spot"),
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

def render_hybrid_gex_table(enriched_df: pd.DataFrame, ticker: str):
    if enriched_df.empty:
        st.warning("No strongest GEX table data available.")
        return

    def highlight_rows(row):
        if row.get("highlight_flag") == "SPOT":
            return ["background-color: rgba(100, 181, 246, 0.25)"] * len(row)
        if row.get("highlight_flag") == "HOT":
            return ["background-color: rgba(255, 215, 64, 0.25)"] * len(row)
        if row.get("auto_flag") in ["A_PLUS_SUPPORT", "A_PLUS_RESISTANCE"]:
            return ["background-color: rgba(102, 187, 106, 0.20)"] * len(row)
        if row.get("auto_flag") in ["TRAP_UP", "TRAP_DOWN", "BREAKOUT_READY", "BREAKDOWN_READY"]:
            return ["background-color: rgba(255, 138, 101, 0.20)"] * len(row)
        if row.get("auto_flag") in ["PRE_BREAKOUT", "PRE_BREAKDOWN"]:
            return ["background-color: rgba(255, 235, 59, 0.18)"] * len(row)
        if row.get("breakout_risk") == "HIGH":
            return ["background-color: rgba(255, 87, 34, 0.12)"] * len(row)
        return [""] * len(row)

    futures_col_name = get_futures_equivalent_label(ticker)

    display_df = enriched_df.copy()

    if "Futures Equivalent" in display_df.columns:
        display_df[futures_col_name] = display_df["Futures Equivalent"]
        display_df = display_df.drop(columns=["Futures Equivalent"])

    if "decision_reason" in display_df.columns:
        display_df["decision_reason"] = display_df["decision_reason"].fillna("").astype(str)

    if "key_interaction" in display_df.columns:
        display_df["key_interaction"] = display_df["key_interaction"].fillna("").astype(str)

    display_cols = [
        "strike",
        futures_col_name,
        "oi_side",
        "gamma_side",
        "agreement",
        "gamma_regime",
        "key_interaction",
        "trigger_state",
        "breakout_risk",
        "trade_score",
        "weighted_gex",
        "weighted_vex",
        "vex_strength",
        "market_behavior",
        "best_trade_type",
        "direction",
        "trade_decision",
        "auto_flag",
        "Entry-Stop-Target",
        "decision_reason",
    ]
    display_cols = [c for c in display_cols if c in display_df.columns]

    styled_df = (
        display_df[display_cols + ["highlight_flag"]]
        .style
        .apply(highlight_rows, axis=1)
        .hide(axis="index")
        .hide(axis="columns", subset=["highlight_flag"])
    )

    st.dataframe(
        styled_df,
        use_container_width=True,
    )



def render_hybrid_scenarios_summary(enriched_df: pd.DataFrame):
    if enriched_df.empty:
        return

    st.write("### Scenario Guide")

    scenarios = [
        ("A_PLUS_SUPPORT", "Strongest long bounce zone. Gamma Key and OI Key align at support."),
        ("A_PLUS_RESISTANCE", "Strongest short rejection zone. Gamma Key and OI Key align at resistance."),
        ("SCALP_ONLY", "Spot is near Gamma Key. Expect pinning/chop. Prefer quick scalps only."),
        ("PRE_BREAKOUT", "OI says resistance but gamma supports the level. Breakout may be building, but price has not confirmed it yet."),
        ("PRE_BREAKDOWN", "OI says support but gamma resists the level. Breakdown may be building, but price has not confirmed it yet."),
        ("TRAP_UP", "Breakout is confirmed through a conflict level. Upside squeeze / expansion is active."),
        ("TRAP_DOWN", "Breakdown is confirmed through a conflict level. Downside expansion is active."),
        ("WATCH_UP", "Conflict level with some energy. Watch for upside resolution."),
        ("WATCH_DOWN", "Conflict level with some energy. Watch for downside resolution."),
        ("BOUNCE_LONG", "Aligned support. Clean long bounce setup."),
        ("BOUNCE_SHORT", "Aligned resistance. Clean short bounce setup."),
        ("FAST_LONG", "Aligned support with high VEX. Favor fast long scalps, not long holds."),
        ("FAST_SHORT", "Aligned resistance with high VEX. Favor fast short scalps, not long holds."),
        ("GAMMA_ONLY_LONG", "Only gamma supports the level. Lower-confidence long."),
        ("GAMMA_ONLY_SHORT", "Only gamma resists the level. Lower-confidence short."),
        ("CHOP_SKIP", "Conflict with low VEX. Messy tape. Best to skip."),
        ("TRANSITION_SKIP", "Near Gamma Flip without enough VEX. Transition zone, avoid forcing trades."),
    ]

    seen_flags = set(enriched_df["auto_flag"].dropna().astype(str).tolist())

    for flag, text in scenarios:
        if flag in seen_flags:
            st.write(f"**{flag}:** {text}")

    if "breakout_risk" in enriched_df.columns:
        if (enriched_df["breakout_risk"] == "HIGH").any():
            st.write("**HIGH breakout risk:** Short-gamma conditions can overwhelm even aligned support/resistance levels.")
        elif (enriched_df["breakout_risk"] == "MEDIUM").any():
            st.write("**MEDIUM breakout risk:** Structure still matters, but short-gamma conditions raise failure risk.")

with tab3:
    st.header("Hybrid View")
    st.write(
        "Hybrid view combines a 16-hour OI price chart with a GEX-by-strike chart "
        "inside one shared subplot figure, so levels line up exactly."
    )

    # Use ONLY the sidebar regression inputs/settings
    spy_a, spy_b = calculate_regression_from_points(
        settings.get("spy_x1", 0.0),
        settings.get("spy_x2", 0.0),
        settings.get("spy_y1", 0.0),
        settings.get("spy_y2", 0.0),
    )

    qqq_a, qqq_b = calculate_regression_from_points(
        settings.get("qqq_x1", 0.0),
        settings.get("qqq_x2", 0.0),
        settings.get("qqq_y1", 0.0),
        settings.get("qqq_y2", 0.0),
    )

    st.subheader("Active Futures Mapping")
    c1, c2 = st.columns(2)

    with c1:
        if spy_a is not None and spy_b is not None:
            st.write(f"**SPY → ES:** a = {spy_a:.6f}, b = {spy_b:.6f}")
        else:
            st.warning("SPY → ES mapping invalid. X1 and X2 cannot be equal.")

    with c2:
        if qqq_a is not None and qqq_b is not None:
            st.write(f"**QQQ → MNQ:** a = {qqq_a:.6f}, b = {qqq_b:.6f}")
        else:
            st.warning("QQQ → MNQ mapping invalid. X1 and X2 cannot be equal.")

    regression_settings = {
        "spy_x1": settings.get("spy_x1", 0.0),
        "spy_x2": settings.get("spy_x2", 0.0),
        "spy_y1": settings.get("spy_y1", 0.0),
        "spy_y2": settings.get("spy_y2", 0.0),
        "qqq_x1": settings.get("qqq_x1", 0.0),
        "qqq_x2": settings.get("qqq_x2", 0.0),
        "qqq_y1": settings.get("qqq_y1", 0.0),
        "qqq_y2": settings.get("qqq_y2", 0.0),
    }

    for ticker in (tickers or DEFAULT_TICKERS):
        st.subheader(f"{ticker} Hybrid View")

        data = ticker_data.get(ticker, {})
        if "error" in data:
            st.error(f"{ticker}: {data['error']}")
            st.divider()
            continue

        try:
            levels_df = data["confluence"]["levels"].copy()
            gamma = data["gamma"]
            oi_key_level = data["oi_payload"].get("key_level")

            hist_full = cached_intraday_history(ticker)
            hist_16h = slice_history_last_hours(hist_full, 16)

            aligned_y_range = get_aligned_y_range(
                hist_df=hist_16h,
                levels_df=levels_df,
                current_spot=float(gamma["spot"]),
            )

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

            st.write("### Strongest GEX Strikes")

            if not curve_df.empty:
                enriched_df = enrich_gex_table(
                    curve_df,
                    levels_df,
                    spot_price=float(gamma["spot"]),
                    ticker=ticker,
                    settings_dict=regression_settings,
                    gamma=gamma,
                    oi_key_level=oi_key_level,
                )

                render_hybrid_gex_table(enriched_df, ticker=ticker)
                render_hybrid_scenarios_summary(enriched_df)
            else:
                st.warning(f"No strongest GEX table data available for {ticker}.")

            st.divider()

        except Exception as e:
            st.error(f"{ticker} hybrid view error: {e}")
            st.divider()
