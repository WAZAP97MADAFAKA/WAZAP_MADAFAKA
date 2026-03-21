import math
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Tuple

import pandas as pd
import yfinance as yf

NY_TZ = ZoneInfo("America/New_York")


def get_spot_price(ticker_symbol: str) -> float:
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period="1d", interval="1m")

    if not hist.empty and "Close" in hist.columns:
        close_series = hist["Close"].dropna()
        if not close_series.empty:
            return float(close_series.iloc[-1])

    fast = getattr(ticker, "fast_info", {})
    last_price = fast.get("lastPrice")
    if last_price is not None:
        return float(last_price)

    raise ValueError(f"Could not get spot price for {ticker_symbol}")


def get_current_spot_price(ticker_symbol: str) -> float:
    return get_spot_price(ticker_symbol)


def get_today_open_spot_price(ticker_symbol: str) -> float:
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period="2d", interval="1m", prepost=False)

    if hist.empty:
        raise ValueError(f"Could not get 1m history for {ticker_symbol}")

    if hist.index.tz is None:
        hist.index = hist.index.tz_localize("UTC")

    hist = hist.tz_convert(NY_TZ)

    today_ny = datetime.now(NY_TZ).date()
    day_data = hist[hist.index.date == today_ny]

    if day_data.empty:
        raise ValueError(f"No intraday data found for {ticker_symbol} for today")

    open_bar = day_data[
        (day_data.index.hour == 9) & (day_data.index.minute == 30)
    ]

    if not open_bar.empty:
        return float(open_bar["Open"].iloc[0])

    after_open = day_data[
        ((day_data.index.hour > 9) | ((day_data.index.hour == 9) & (day_data.index.minute >= 30)))
    ]

    if after_open.empty:
        raise ValueError(f"No 9:30 or later bar found for {ticker_symbol}")

    return float(after_open["Close"].iloc[0])


def get_first_n_expirations(ticker_symbol: str, n: int) -> List[str]:
    ticker = yf.Ticker(ticker_symbol)
    expirations = list(ticker.options)

    if not expirations:
        raise ValueError(f"No expirations found for {ticker_symbol}")

    if len(expirations) < n:
        print(f"Warning: only found {len(expirations)} expirations for {ticker_symbol}")

    return expirations[:n]


def get_weighted_option_data(
    ticker_symbol: str,
    weights: List[float]
) -> Tuple[float, List[str], pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        spot
        expirations
        combined_calls
        combined_puts

    combined_calls / combined_puts columns:
        strike
        total_open_interest
        weighted_open_interest
        total_volume
        weighted_volume
        avg_implied_volatility
    """
    ticker = yf.Ticker(ticker_symbol)
    spot = get_spot_price(ticker_symbol)
    expirations = get_first_n_expirations(ticker_symbol, len(weights))

    all_calls = []
    all_puts = []

    for exp, weight in zip(expirations, weights):
        chain = ticker.option_chain(exp)

        calls = chain.calls.copy()
        puts = chain.puts.copy()

        for df in (calls, puts):
            df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
            df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce").fillna(0)
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
            df["impliedVolatility"] = pd.to_numeric(df["impliedVolatility"], errors="coerce").fillna(0)

        calls["weightedOI"] = calls["openInterest"] * weight
        puts["weightedOI"] = puts["openInterest"] * weight

        calls["weightedVolume"] = calls["volume"] * weight
        puts["weightedVolume"] = puts["volume"] * weight

        calls["expiration"] = exp
        puts["expiration"] = exp
        calls["weight"] = weight
        puts["weight"] = weight

        all_calls.append(
            calls[[
                "strike", "openInterest", "weightedOI",
                "volume", "weightedVolume", "impliedVolatility",
                "expiration", "weight"
            ]]
        )
        all_puts.append(
            puts[[
                "strike", "openInterest", "weightedOI",
                "volume", "weightedVolume", "impliedVolatility",
                "expiration", "weight"
            ]]
        )

    calls_df = pd.concat(all_calls, ignore_index=True)
    puts_df = pd.concat(all_puts, ignore_index=True)

    combined_calls = (
        calls_df.groupby("strike", as_index=False)
        .agg(
            total_open_interest=("openInterest", "sum"),
            weighted_open_interest=("weightedOI", "sum"),
            total_volume=("volume", "sum"),
            weighted_volume=("weightedVolume", "sum"),
            avg_implied_volatility=("impliedVolatility", "mean"),
        )
        .sort_values("strike")
        .reset_index(drop=True)
    )

    combined_puts = (
        puts_df.groupby("strike", as_index=False)
        .agg(
            total_open_interest=("openInterest", "sum"),
            weighted_open_interest=("weightedOI", "sum"),
            total_volume=("volume", "sum"),
            weighted_volume=("weightedVolume", "sum"),
            avg_implied_volatility=("impliedVolatility", "mean"),
        )
        .sort_values("strike")
        .reset_index(drop=True)
    )

    return spot, expirations, combined_calls, combined_puts


def get_local_range(spot: float, max_distance: float) -> Tuple[float, float]:
    return spot - max_distance, spot + max_distance


def filter_local_calls(combined_calls: pd.DataFrame, spot: float, max_distance: float) -> pd.DataFrame:
    _, max_strike = get_local_range(spot, max_distance)
    return combined_calls[
        (combined_calls["strike"] > spot) &
        (combined_calls["strike"] <= max_strike)
    ].copy()


def filter_local_puts(combined_puts: pd.DataFrame, spot: float, max_distance: float) -> pd.DataFrame:
    min_strike, _ = get_local_range(spot, max_distance)
    return combined_puts[
        (combined_puts["strike"] < spot) &
        (combined_puts["strike"] >= min_strike)
    ].copy()


def choose_nearest_key_level(levels_df: pd.DataFrame, spot: float, score_column: str) -> float | None:
    if levels_df.empty:
        return None

    df = levels_df.copy()
    df["distance_to_spot"] = (df["strike"] - spot).abs()
    df = df.sort_values(
        by=["distance_to_spot", score_column],
        ascending=[True, False]
    )
    return float(df.iloc[0]["strike"])


def time_to_expiration_in_years(expiration: str) -> float:
    exp_dt = datetime.strptime(expiration, "%Y-%m-%d")
    now = datetime.now()
    exp_close = exp_dt.replace(hour=16, minute=0, second=0, microsecond=0)
    seconds = (exp_close - now).total_seconds()
    return max(seconds, 60) / (365.0 * 24 * 60 * 60)


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def bs_gamma(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0

    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        return norm_pdf(d1) / (S * sigma * math.sqrt(T))
    except Exception:
        return 0.0


def market_hours_now() -> bool:
    now = datetime.now(NY_TZ)
    return now.weekday() < 5 and ((now.hour > 9 or (now.hour == 9 and now.minute >= 30)) and now.hour < 16)