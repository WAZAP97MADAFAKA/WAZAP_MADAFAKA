import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta

# ================================
# CONFIG
# ================================
st.set_page_config(layout="wide")
st.title("Options Dashboard 4")

TICKERS = ["SPY", "QQQ"]

# ================================
# AUTO REFRESH (1 min)
# ================================
st_autorefresh = st.experimental_rerun

# ================================
# YFINANCE DATA (FIXED)
# ================================
def _download_yf_intraday(ticker):
    try:
        df = yf.download(
            tickers=ticker,
            period="2d",
            interval="1m",
            prepost=True,
            progress=False,
        )

        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        })

        df.index = pd.to_datetime(df.index)

        # ✅ REMOVE WEEKENDS
        df = df[df.index.dayofweek < 5]

        # ✅ LAST 24H ONLY
        cutoff = df.index.max() - pd.Timedelta(hours=24)
        df = df[df.index >= cutoff]

        return df

    except:
        return pd.DataFrame()

# ================================
# FAKE DATA PLACEHOLDER (replace with Polygon output)
# ================================
def get_mock_gamma():
    strikes = np.arange(630, 690, 1)
    gex = np.random.randn(len(strikes)) * 1e8

    return {
        "gex_curve": [{"strike": s, "weighted_gex": g} for s, g in zip(strikes, gex)],
        "top_supports": [{"strike": 650}],
        "top_resistances": [{"strike": 660}],
        "key_level": 660,
        "gamma_flip": 658,
        "spot": 657
    }

def get_mock_oi():
    return {
        "key_level": 657
    }

# ================================
# GEX BAR CHART (UPDATED)
# ================================
def build_gex_bar_chart(ticker, gamma, oi_key_level):
    curve = pd.DataFrame(gamma["gex_curve"])
    curve = curve.sort_values("strike")

    support_strikes = {x["strike"] for x in gamma["top_supports"]}
    resistance_strikes = {x["strike"] for x in gamma["top_resistances"]}

    def classify_sr(s):
        if s in support_strikes:
            return "SUPPORT"
        if s in resistance_strikes:
            return "RESISTANCE"
        return "OTHER"

    curve["sr_type"] = curve["strike"].apply(classify_sr)

    colors = ["green" if x >= 0 else "red" for x in curve["weighted_gex"]]

    fig = go.Figure()

    fig.add_bar(
        x=curve["strike"],
        y=curve["weighted_gex"],
        marker_color=colors
    )

    y_max = curve["weighted_gex"].max()
    y_min = curve["weighted_gex"].min()
    y_range = y_max - y_min

    # Lines
    fig.add_vline(x=gamma["key_level"], line_dash="dash", line_color="yellow")
    fig.add_vline(x=oi_key_level, line_dash="dot", line_color="blue")
    fig.add_vline(x=gamma["gamma_flip"], line_dash="longdash", line_color="orange")
    fig.add_vline(x=gamma["spot"], line_color="white")

    # Labels
    fig.add_annotation(x=gamma["key_level"], y=y_max, text="Gamma Key", showarrow=False)
    fig.add_annotation(x=oi_key_level, y=y_max*0.9, text="OI Key", showarrow=False)
    fig.add_annotation(x=gamma["gamma_flip"], y=y_max*0.8, text="Flip", showarrow=False)
    fig.add_annotation(x=gamma["spot"], y=y_max*0.7, text="Spot", showarrow=False)

    # R / S labels
    for _, row in curve.iterrows():
        if row["sr_type"] == "SUPPORT":
            txt = "S"
            color = "green"
        elif row["sr_type"] == "RESISTANCE":
            txt = "R"
            color = "red"
        else:
            continue

        fig.add_annotation(
            x=row["strike"],
            y=row["weighted_gex"],
            text=txt,
            showarrow=False,
            font=dict(color=color)
        )

    curve["abs_weighted_gex"] = curve["weighted_gex"].abs()
    curve = curve.sort_values("abs_weighted_gex", ascending=False)

    return fig, curve

# ================================
# TABS
# ================================
tab1, tab2, tab3 = st.tabs(["Dashboard", "Charts", "GEX Chart"])

# ================================
# DASHBOARD
# ================================
with tab1:
    st.subheader("Dashboard")

    for ticker in TICKERS:
        gamma = get_mock_gamma()
        oi = get_mock_oi()

        st.write(f"### {ticker}")
        st.write("Spot:", gamma["spot"])
        st.write("Gamma Flip:", gamma["gamma_flip"])
        st.write("Gamma Key:", gamma["key_level"])
        st.write("OI Key:", oi["key_level"])

# ================================
# CHARTS
# ================================
with tab2:
    st.subheader("Charts (Last 24h - No Weekends)")

    for ticker in TICKERS:
        df = _download_yf_intraday(ticker)

        if df.empty:
            st.warning(f"{ticker}: No data")
            continue

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["close"],
            name="Price"
        ))

        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

# ================================
# GEX CHART
# ================================
with tab3:
    st.subheader("GEX Chart")

    for ticker in TICKERS:
        gamma = get_mock_gamma()
        oi = get_mock_oi()

        fig, curve = build_gex_bar_chart(ticker, gamma, oi["key_level"])

        st.plotly_chart(fig, use_container_width=True)

        st.write("### Strongest GEX Strikes")

        st.dataframe(
            curve[["strike", "sr_type", "weighted_gex", "abs_weighted_gex"]].head(20),
            use_container_width=True,
            hide_index=True
        )