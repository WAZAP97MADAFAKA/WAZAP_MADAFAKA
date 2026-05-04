import json
import math
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
from options_common import get_intraday_history_last_24h_extended


if "POLYGON_API_KEY" in st.secrets and "POLYGON_API_KEY" not in os.environ:
    os.environ["POLYGON_API_KEY"] = st.secrets["POLYGON_API_KEY"]
if "MASSIVE_API_KEY" in st.secrets and "MASSIVE_API_KEY" not in os.environ:
    os.environ["MASSIVE_API_KEY"] = st.secrets["MASSIVE_API_KEY"]

st.set_page_config(page_title="Options Dashboard 4", layout="wide")
st.title("Options Dashboard 4")
st.caption("Polygon/Massive-based OI + GEX + DEX + VEX dashboard")

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
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
    if hist_df is None or hist_df.empty:
        return pd.DataFrame(columns=["datetime", "close"])

    df = hist_df.copy()
    if "datetime" not in df.columns:
        return df

    df = df.sort_values("datetime").reset_index(drop=True)
    end_time = df["datetime"].max()
    start_time = end_time - pd.Timedelta(hours=hours)
    return df[df["datetime"] >= start_time].copy().reset_index(drop=True)


def get_visual_strike_step(ticker: str, gamma: dict | None = None) -> float:
    if gamma is not None:
        strike_step = gamma.get("strike_step")
        if strike_step is not None:
            try:
                strike_step = float(strike_step)
                if strike_step > 0:
                    return strike_step
            except Exception:
                pass

    ticker = str(ticker).upper()
    if ticker == "SPX":
        return 5.0
    if ticker == "NDX":
        return 10.0
    return 1.0


def get_curve_df(gamma: dict) -> pd.DataFrame:
    curve = pd.DataFrame(gamma.get("gex_curve", []))
    if curve.empty:
        curve = pd.DataFrame(gamma.get("gex_curve_wide", []))

    needed = ["strike", "weighted_gex", "weighted_dex", "weighted_vex", "weighted_open_interest"]
    if curve.empty:
        return pd.DataFrame(columns=needed)

    for col in needed:
        if col not in curve.columns:
            curve[col] = 0.0
        curve[col] = pd.to_numeric(curve[col], errors="coerce").fillna(0.0)

    if "total_open_interest" not in curve.columns:
        curve["total_open_interest"] = curve["weighted_open_interest"]
    curve["total_open_interest"] = pd.to_numeric(curve["total_open_interest"], errors="coerce").fillna(0.0)

    if "call_weighted_oi" not in curve.columns:
        curve["call_weighted_oi"] = 0.0
    if "put_weighted_oi" not in curve.columns:
        curve["put_weighted_oi"] = 0.0

    curve["call_weighted_oi"] = pd.to_numeric(curve["call_weighted_oi"], errors="coerce").fillna(0.0)
    curve["put_weighted_oi"] = pd.to_numeric(curve["put_weighted_oi"], errors="coerce").fillna(0.0)

    return curve.dropna(subset=["strike"]).sort_values("strike").reset_index(drop=True)


def get_aligned_y_range(hist_df: pd.DataFrame, curve_df: pd.DataFrame, current_spot: float, strike_step: float):
    values = []

    if curve_df is not None and not curve_df.empty and "strike" in curve_df.columns:
        values.extend(pd.to_numeric(curve_df["strike"], errors="coerce").dropna().tolist())

    if hist_df is not None and not hist_df.empty and "close" in hist_df.columns:
        values.extend(pd.to_numeric(hist_df["close"], errors="coerce").dropna().tolist())

    if current_spot is not None:
        values.append(float(current_spot))

    if not values:
        return None

    step = float(strike_step or 1.0)
    lower = min(values)
    upper = max(values)
    pad = step if upper <= lower else max(step, (upper - lower) * 0.05)

    y_min = step * math.floor((lower - pad) / step)
    y_max = step * math.ceil((upper + pad) / step)
    return [round(y_min, 2), round(y_max, 2)]


def get_shared_yaxis_config(forced_y_range, strike_step: float = 1.0):
    if forced_y_range is None:
        return {"fixedrange": False}

    step = float(strike_step or 1.0)
    y_min, _ = forced_y_range
    tick0 = step * math.floor(float(y_min) / step)
    return {
        "range": forced_y_range,
        "tickmode": "linear",
        "tick0": tick0,
        "dtick": step,
        "fixedrange": False,
    }


def compute_net_vex_ratio(net_vex: float, abs_vex: float) -> float:
    if abs_vex <= 0:
        return 0.0
    return max(-1.0, min(1.0, float(net_vex) / float(abs_vex)))


def classify_abs_vex_regime(abs_vex: float, ticker: str) -> str:
    if ticker == "SPY":
        if abs_vex < 8000:
            return "LOW"
        if abs_vex <= 25000:
            return "MEDIUM"
        return "HIGH"

    if ticker == "QQQ":
        if abs_vex < 10000:
            return "LOW"
        if abs_vex <= 30000:
            return "MEDIUM"
        return "HIGH"

    if abs_vex < 8000:
        return "LOW"
    if abs_vex <= 25000:
        return "MEDIUM"
    return "HIGH"


def classify_net_vex_bias(net_vex_ratio: float) -> str:
    if net_vex_ratio <= -0.50:
        return "STRONG NEGATIVE"
    if net_vex_ratio < -0.15:
        return "NEGATIVE"
    if net_vex_ratio <= 0.15:
        return "NEUTRAL"
    if net_vex_ratio < 0.50:
        return "POSITIVE"
    return "STRONG POSITIVE"


def update_vex_history(ticker: str, timestamp, abs_vex: float, net_vex: float, max_points: int = 120):
    key = f"vex_history_{ticker}"
    if key not in st.session_state:
        st.session_state[key] = []

    history = st.session_state[key]
    history.append(
        {
            "datetime": pd.to_datetime(timestamp),
            "abs_vex": float(abs_vex),
            "net_vex": float(net_vex),
            "net_vex_ratio": compute_net_vex_ratio(net_vex, abs_vex),
        }
    )

    if len(history) > max_points:
        history = history[-max_points:]

    st.session_state[key] = history
    return pd.DataFrame(history)


def compute_ves_signal(vex_history_df: pd.DataFrame, ticker: str, gamma_regime_payload: str | None = None):
    if vex_history_df is None or vex_history_df.empty:
        return {"state": "OFF", "reason": "No VEX history yet."}

    df = vex_history_df.sort_values("datetime").reset_index(drop=True)
    current = df.iloc[-1]
    current_abs_vex = float(current["abs_vex"])
    current_net_ratio = float(current["net_vex_ratio"])
    current_abs_regime = classify_abs_vex_regime(current_abs_vex, ticker)

    if len(df) < 3:
        if current_abs_regime == "HIGH" and abs(current_net_ratio) >= 0.35:
            return {"state": "ON", "reason": "High Abs VEX with strong Net VEX bias."}
        return {"state": "OFF", "reason": "Not enough VEX history yet."}

    lookback_df = df.iloc[-6:-1] if len(df) >= 6 else df.iloc[:-1]
    prev_abs_mean = float(lookback_df["abs_vex"].mean()) if not lookback_df.empty else current_abs_vex
    prev_ratio_mean = float(lookback_df["net_vex_ratio"].mean()) if not lookback_df.empty else current_net_ratio

    abs_vex_acceleration = current_abs_vex >= (prev_abs_mean * 1.15 if prev_abs_mean > 0 else current_abs_vex)
    net_ratio_acceleration = abs(current_net_ratio) >= abs(prev_ratio_mean) + 0.10
    strong_bias = abs(current_net_ratio) >= 0.35
    very_strong_bias = abs(current_net_ratio) >= 0.50

    gamma_regime_payload = str(gamma_regime_payload or "")
    short_gamma_proxy = "SHORT" in gamma_regime_payload
    long_gamma_proxy = "LONG" in gamma_regime_payload

    ves_on = False
    reason = "No expansion signal."

    if current_abs_regime == "HIGH" and strong_bias:
        ves_on = True
        reason = "High Abs VEX with directional Net VEX bias."
    elif current_abs_regime == "MEDIUM" and abs_vex_acceleration and strong_bias:
        ves_on = True
        reason = "Abs VEX is accelerating with directional Net VEX bias."
    elif short_gamma_proxy and current_abs_regime in ["MEDIUM", "HIGH"] and abs_vex_acceleration and net_ratio_acceleration:
        ves_on = True
        reason = "Short-gamma regime with rising VEX pressure."
    elif very_strong_bias and abs_vex_acceleration:
        ves_on = True
        reason = "Directional Net VEX is strong and VEX is building."

    if long_gamma_proxy and current_abs_regime == "LOW":
        ves_on = False
        reason = "Long-gamma and low-energy environment."

    return {"state": "ON" if ves_on else "OFF", "reason": reason}


def dex_ratio_to_color(ratio: float) -> str:
    ratio = max(-1.0, min(1.0, float(ratio or 0.0)))

    # Positive DEX = Deep Blue. Zero = White. Negative DEX = Deep Red.
    if ratio >= 0:
        r = int(255 * (1 - ratio))
        g = int(255 * (1 - ratio))
        b = 255
    else:
        strength = abs(ratio)
        r = 255
        g = int(255 * (1 - strength))
        b = int(255 * (1 - strength))

    return f"rgb({r},{g},{b})"


def render_metrics_panel(
    net_dex: float,
    net_gex: float,
    net_vex: float,
    abs_vex: float,
    abs_vex_regime: str,
    net_vex_bias: str,
    ves_signal: dict,
):
    st.write("### GEX / DEX / VEX")
    c1, c2, c3 = st.columns(3)
    c1.metric("Net DEX", f"{net_dex:,.0f}")
    c2.metric("Net GEX", f"{net_gex:,.0f}")
    c3.metric("Net VEX", f"{net_vex:,.0f}")

    st.write("### VEX & VES")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Abs VEX", f"{abs_vex:,.0f}")
    c2.metric("Abs VEX Regime", abs_vex_regime)
    c3.metric("Net VEX Bias", net_vex_bias)
    c4.metric("VES", ves_signal.get("state", "OFF"))

    if ves_signal.get("state") == "ON":
        st.warning(f"VES ON — {ves_signal.get('reason', '')}")
    else:
        st.info(f"VES OFF — {ves_signal.get('reason', '')}")


def add_wall_lines(fig, y_value, label, color, rows_cols):
    if y_value is None or pd.isna(y_value):
        return

    y_value = float(y_value)
    for row, col, yref in rows_cols:
        fig.add_hline(
            y=y_value,
            line_width=2.5,
            line_dash="dash",
            line_color=color,
            row=row,
            col=col,
        )
        fig.add_annotation(
            xref="paper",
            yref=yref,
            x=0.98 if col != 1 else 0.30,
            y=y_value,
            text=f"{label} {y_value:.2f}",
            showarrow=False,
            font=dict(color=color, size=11),
            bgcolor="rgba(0,0,0,0.40)",
            xanchor="right",
            yanchor="middle",
        )


def build_hybrid_subplot_figure(
    ticker,
    hist_df,
    gamma,
    vex_history_df=None,
    forced_y_range=None,
):
    fig = make_subplots(
        rows=2,
        cols=3,
        shared_yaxes=False,
        horizontal_spacing=0.025,
        vertical_spacing=0.08,
        row_heights=[0.72, 0.28],
        column_widths=[0.54, 0.25, 0.21],
        specs=[
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "xy", "colspan": 3, "secondary_y": True}, None, None],
        ],
    )

    current_spot = float(gamma["spot"])
    strike_step = get_visual_strike_step(ticker, gamma)
    curve = get_curve_df(gamma)

    if curve.empty:
        return fig, pd.DataFrame()

    dex_abs_sum = float(curve["weighted_dex"].abs().sum())
    if dex_abs_sum <= 0:
        curve["dex_ratio"] = 0.0
    else:
        curve["dex_ratio"] = (curve["weighted_dex"] / dex_abs_sum).clip(-1, 1)

    # -----------------------------
    # Top-left: price + DEX by strike
    # -----------------------------
    fig.add_trace(
        go.Scatter(
            x=hist_df["datetime"],
            y=hist_df["close"],
            mode="lines",
            name=f"{ticker} Price",
            line=dict(width=2, color="#90CAF9"),
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
                windows = [
                    (current_day, current_day + timedelta(hours=4), "rgba(80, 80, 120, 0.12)"),
                    (current_day + timedelta(hours=4), current_day + timedelta(hours=9, minutes=30), "rgba(70, 120, 180, 0.12)"),
                    (current_day + timedelta(hours=16), current_day + timedelta(hours=20), "rgba(150, 90, 170, 0.12)"),
                    (current_day + timedelta(hours=20), current_day + timedelta(days=1), "rgba(80, 80, 120, 0.12)"),
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

    # DEX lines on price chart. OI level lines are intentionally removed.
    for _, row in curve.iterrows():
        strike = float(row["strike"])
        dex_ratio = float(row.get("dex_ratio", 0.0))
        dex_value = float(row.get("weighted_dex", 0.0))
        color = dex_ratio_to_color(dex_ratio)
        width = max(1.0, min(7.0, 1.0 + abs(dex_ratio) * 80.0))

        fig.add_hline(
            y=strike,
            line_color=color,
            line_width=width,
            line_dash="solid",
            opacity=max(0.25, min(1.0, 0.35 + abs(dex_ratio) * 12.0)),
            row=1,
            col=1,
        )

        fig.add_annotation(
            xref="paper",
            yref="y",
            x=0.01,
            y=strike,
            text=f"DEX {dex_value:,.0f}",
            showarrow=False,
            font=dict(size=9, color=color),
            bgcolor="rgba(0,0,0,0.25)",
            xanchor="left",
            yanchor="middle",
        )

    fig.add_hline(
        y=current_spot,
        line_color="#64B5F6",
        line_width=1.6,
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
        yanchor="bottom",
    )

    # -----------------------------
    # Top-middle: GEX by strike
    # -----------------------------
    gex_colors = ["#00C853" if float(v) >= 0 else "#D50000" for v in curve["weighted_gex"]]
    fig.add_trace(
        go.Bar(
            x=curve["weighted_gex"],
            y=curve["strike"],
            orientation="h",
            marker_color=gex_colors,
            name="Weighted GEX",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Keep Gamma Key / Gamma Flip on GEX chart.
    gamma_key_local = gamma.get("gamma_key_local", gamma.get("key_level"))
    gamma_key_global = gamma.get("gamma_key_global")
    gamma_flip = gamma.get("gamma_flip")

    if gamma_key_local is not None:
        fig.add_hline(y=float(gamma_key_local), line_width=2, line_dash="dash", line_color="#BA68C8", row=1, col=2)
        fig.add_annotation(xref="paper", yref="y2", x=0.78, y=float(gamma_key_local), text=f"Gamma Key Local {float(gamma_key_local):.2f}", showarrow=False, font=dict(color="#BA68C8", size=10), bgcolor="rgba(0,0,0,0.35)", xanchor="right", yanchor="bottom")

    if gamma_key_global is not None:
        fig.add_hline(y=float(gamma_key_global), line_width=2, line_dash="dot", line_color="#CE93D8", row=1, col=2)
        fig.add_annotation(xref="paper", yref="y2", x=0.78, y=float(gamma_key_global), text=f"Gamma Key Global {float(gamma_key_global):.2f}", showarrow=False, font=dict(color="#CE93D8", size=10), bgcolor="rgba(0,0,0,0.35)", xanchor="right", yanchor="top")

    if gamma_flip is not None:
        fig.add_hline(y=float(gamma_flip), line_width=2, line_dash="longdash", line_color="#FF9800", row=1, col=2)
        fig.add_annotation(xref="paper", yref="y2", x=0.78, y=float(gamma_flip), text=f"Gamma Flip {float(gamma_flip):.2f}", showarrow=False, font=dict(color="#FF9800", size=10), bgcolor="rgba(0,0,0,0.35)", xanchor="right", yanchor="middle")

    # -----------------------------
    # Top-right: OI by strike
    # Call OI = positive green
    # Put OI = negative red
    # Net OI = Call OI + Put OI, where Put OI is negative
    # Net OI is plotted as BLUE markers so it cannot visually stack on top of Call OI.
    # -----------------------------
    for required_col in ["call_weighted_oi", "put_weighted_oi"]:
        if required_col not in curve.columns:
            curve[required_col] = 0.0

    curve["call_weighted_oi"] = pd.to_numeric(
        curve["call_weighted_oi"], errors="coerce"
    ).fillna(0.0)

    curve["put_weighted_oi"] = pd.to_numeric(
        curve["put_weighted_oi"], errors="coerce"
    ).fillna(0.0)

    # IMPORTANT:
    # Calls are positive.
    # Puts are negative.
    # Net OI = Call OI + signed Put OI.
    curve["call_oi_positive"] = curve["call_weighted_oi"].abs()
    curve["put_oi_negative"] = -curve["put_weighted_oi"].abs()
    curve["net_weighted_oi"] = curve["call_oi_positive"] + curve["put_oi_negative"]

    # 1) Net OI FIRST as blue markers.
    # This avoids the visual problem where Net OI looks stacked onto Call OI.
    fig.add_trace(
        go.Scatter(
            x=curve["net_weighted_oi"],
            y=curve["strike"],
            mode="markers",
            marker=dict(
                color="#42A5F5",
                size=8,
                line=dict(color="#0D47A1", width=1),
            ),
            name="Net OI (Call + Put)",
            hovertemplate=(
                "Strike %{y}<br>"
                "Net OI %{x:,.0f}<br>"
                "Call OI %{customdata[0]:,.0f}<br>"
                "Put OI %{customdata[1]:,.0f}<extra></extra>"
            ),
            customdata=curve[["call_oi_positive", "put_oi_negative"]].to_numpy(),
            showlegend=True,
        ),
        row=1,
        col=3,
    )

    # 2) Call OI positive green bars from zero.
    fig.add_trace(
        go.Bar(
            x=curve["call_oi_positive"],
            y=curve["strike"],
            base=0,
            orientation="h",
            marker_color="#00C853",
            name="Call OI (+)",
            opacity=0.78,
            hovertemplate="Strike %{y}<br>Call OI +%{x:,.0f}<extra></extra>",
            showlegend=True,
        ),
        row=1,
        col=3,
    )

    # 3) Put OI negative red bars from zero.
    fig.add_trace(
        go.Bar(
            x=curve["put_oi_negative"],
            y=curve["strike"],
            base=0,
            orientation="h",
            marker_color="#D50000",
            name="Put OI (-)",
            opacity=0.78,
            hovertemplate="Strike %{y}<br>Put OI %{x:,.0f}<extra></extra>",
            showlegend=True,
        ),
        row=1,
        col=3,
    )

    fig.add_vline(
        x=0,
        line_width=1,
        line_dash="solid",
        line_color="rgba(255,255,255,0.45)",
        row=1,
        col=3,
    )

    oi_axis_max = max(
        float(curve["call_oi_positive"].abs().max()) if not curve.empty else 0.0,
        float(curve["put_oi_negative"].abs().max()) if not curve.empty else 0.0,
        float(curve["net_weighted_oi"].abs().max()) if not curve.empty else 0.0,
        1.0,
    )

    # Call Wall / Put Wall on all three top charts.
    call_wall = gamma.get("call_wall")
    put_wall = gamma.get("put_wall")
    add_wall_lines(
        fig,
        call_wall,
        "Call Wall",
        "#FFD600",
        rows_cols=[(1, 1, "y"), (1, 2, "y2"), (1, 3, "y3")],
    )
    add_wall_lines(
        fig,
        put_wall,
        "Put Wall",
        "#FF9800",
        rows_cols=[(1, 1, "y"), (1, 2, "y2"), (1, 3, "y3")],
    )

    # -----------------------------
    # Bottom: Abs VEX + Net VEX Ratio
    # -----------------------------
    if vex_history_df is not None and not vex_history_df.empty:
        vex_history_df = vex_history_df.sort_values("datetime").reset_index(drop=True)

        fig.add_trace(
            go.Scatter(
                x=vex_history_df["datetime"],
                y=vex_history_df["abs_vex"],
                mode="lines",
                name="Abs VEX",
                line=dict(width=2, color="#80CBC4"),
            ),
            row=2,
            col=1,
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=vex_history_df["datetime"],
                y=vex_history_df["net_vex_ratio"],
                mode="lines",
                name="Net VEX Ratio",
                line=dict(width=2, dash="dot", color="#CE93D8"),
            ),
            row=2,
            col=1,
            secondary_y=True,
        )

        if ticker == "SPY":
            low_band = 8000
            med_band = 25000
        elif ticker == "QQQ":
            low_band = 10000
            med_band = 30000
        else:
            low_band = 8000
            med_band = 25000

        fig.add_hrect(y0=0, y1=low_band, fillcolor="rgba(0, 200, 83, 0.08)", line_width=0, row=2, col=1)
        fig.add_hrect(y0=low_band, y1=med_band, fillcolor="rgba(255, 193, 7, 0.08)", line_width=0, row=2, col=1)
        fig.add_hrect(
            y0=med_band,
            y1=max(float(vex_history_df["abs_vex"].max()) * 1.15, med_band + 1),
            fillcolor="rgba(244, 67, 54, 0.08)",
            line_width=0,
            row=2,
            col=1,
        )

    shared_yaxis = get_shared_yaxis_config(forced_y_range, strike_step=strike_step)

    fig.update_yaxes(**shared_yaxis, showticklabels=True, row=1, col=1)
    fig.update_yaxes(**shared_yaxis, showticklabels=True, matches="y", row=1, col=2)
    fig.update_yaxes(**shared_yaxis, showticklabels=True, matches="y", row=1, col=3)

    fig.update_xaxes(title_text="Time", rangebreaks=[dict(bounds=["sat", "mon"])], rangeslider=dict(visible=False), row=1, col=1)
    fig.update_xaxes(title_text="Weighted GEX", row=1, col=2)
    fig.update_xaxes(
        title_text="Weighted OI: Call + / Put - / Net Marker",
        range=[-oi_axis_max * 1.15, oi_axis_max * 1.15],
        zeroline=True,
        zerolinecolor="rgba(255,255,255,0.45)",
        row=1,
        col=3,
    )
    fig.update_xaxes(title_text="Time", rangebreaks=[dict(bounds=["sat", "mon"])], rangeslider=dict(visible=False), row=2, col=1)

    fig.update_yaxes(title_text="Price / Strike", row=1, col=1)
    fig.update_yaxes(title_text="Strike", row=1, col=2)
    fig.update_yaxes(title_text="Strike", row=1, col=3)
    fig.update_yaxes(title_text="Abs VEX", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Net VEX Ratio", row=2, col=1, secondary_y=True, range=[-1, 1], showgrid=False)

    fig.update_layout(
        title=f"{ticker} - Price / GEX / OI View",
        template="plotly_dark",
        height=950,
        margin=dict(l=60, r=110, t=70, b=50),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        barmode="overlay",
    )

    return fig, curve


settings = load_settings()

st.sidebar.header("Settings")

ticker_options = ["SPY", "QQQ", "SPX", "NDX"]
tickers = st.sidebar.multiselect(
    "Tickers",
    options=ticker_options,
    default=settings["tickers"] if settings["tickers"] else DEFAULT_TICKERS,
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
        "max_distance": float(max_distance),
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

st.header("Hybrid View")
st.write(
    "Price, GEX, and OI are aligned by strike. DEX is displayed on the price chart by strike using a red-white-blue intensity scale."
)

for ticker in (tickers or DEFAULT_TICKERS):
    st.subheader(f"{ticker} Hybrid View")

    oi_path = os.path.join(DATA_CACHE_DIR, f"oi_{ticker}.json")
    oi_payload = load_json(oi_path, {})

    if not oi_payload:
        st.error(f"{ticker}: No OI cache found yet. Run the morning refresh first.")
        st.divider()
        continue

    try:
        gamma = cached_gamma(
            ticker,
            tuple(weights),
            float(max_distance),
            int(num_levels),
            oi_payload.get("oi_fixed_spot"),
        )

        curve_df = get_curve_df(gamma)
        if curve_df.empty:
            st.warning(f"{ticker}: No curve data available.")
            st.divider()
            continue

        net_gex = float(gamma.get("total_net_gex", curve_df["weighted_gex"].sum()))
        net_dex = float(gamma.get("total_net_dex", curve_df["weighted_dex"].sum()))
        net_vex = float(gamma.get("total_net_vex", curve_df["weighted_vex"].sum()))
        abs_vex = float(curve_df["weighted_vex"].abs().sum())
        net_vex_ratio = compute_net_vex_ratio(net_vex, abs_vex)
        abs_vex_regime = classify_abs_vex_regime(abs_vex, ticker)
        net_vex_bias = classify_net_vex_bias(net_vex_ratio)

        vex_history_df = update_vex_history(
            ticker=ticker,
            timestamp=pd.Timestamp.now(),
            abs_vex=abs_vex,
            net_vex=net_vex,
        )

        ves_signal = compute_ves_signal(
            vex_history_df=vex_history_df,
            ticker=ticker,
            gamma_regime_payload=gamma.get("regime"),
        )

        render_metrics_panel(
            net_dex=net_dex,
            net_gex=net_gex,
            net_vex=net_vex,
            abs_vex=abs_vex,
            abs_vex_regime=abs_vex_regime,
            net_vex_bias=net_vex_bias,
            ves_signal=ves_signal,
        )

        hist_full = cached_intraday_history(ticker)
        hist_8h = slice_history_last_hours(hist_full, 8)
        strike_step = get_visual_strike_step(ticker, gamma)

        aligned_y_range = get_aligned_y_range(
            hist_df=hist_8h,
            curve_df=curve_df,
            current_spot=float(gamma["spot"]),
            strike_step=strike_step,
        )

        hybrid_fig, _ = build_hybrid_subplot_figure(
            ticker=ticker,
            hist_df=hist_8h,
            gamma=gamma,
            vex_history_df=vex_history_df,
            forced_y_range=aligned_y_range,
        )

        st.plotly_chart(
            hybrid_fig,
            use_container_width=True,
            key=f"{ticker}_hybrid_price_gex_oi_chart",
        )

        with st.expander(f"{ticker} Curve Data"):
            display_cols = [
                "strike",
                "weighted_dex",
                "dex_ratio",
                "weighted_gex",
                "weighted_vex",
                "weighted_open_interest",
                "call_weighted_oi",
                "put_weighted_oi",
            ]
            display_cols = [c for c in display_cols if c in curve_df.columns]
            st.dataframe(curve_df[display_cols].sort_values("strike", ascending=False), use_container_width=True)

        st.divider()

    except Exception as e:
        st.error(f"{ticker} hybrid view error: {e}")
        st.divider()
