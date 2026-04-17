import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from massive import RESTClient

from options_config import NY_TIMEZONE

NY_TZ = ZoneInfo(NY_TIMEZONE)


def get_api_key() -> str:
    api_key = os.getenv("POLYGON_API_KEY") or os.getenv("MASSIVE_API_KEY")
    if not api_key:
        raise ValueError("Missing POLYGON_API_KEY or MASSIVE_API_KEY.")
    return api_key


def get_client() -> RESTClient:
    return RESTClient(api_key=get_api_key())


def obj_to_dict(obj):
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [obj_to_dict(x) for x in obj]
    if isinstance(obj, tuple):
        return [obj_to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: obj_to_dict(v) for k, v in obj.items()}
    if hasattr(obj, "model_dump"):
        return obj_to_dict(obj.model_dump())
    if hasattr(obj, "dict"):
        return obj_to_dict(obj.dict())
    if hasattr(obj, "__dict__"):
        return {k: obj_to_dict(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return obj


def _download_yf_intraday(
    ticker_symbol: str,
    period: str,
    interval: str,
    prepost: bool,
) -> pd.DataFrame:
    df = yf.download(
        tickers=ticker_symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        prepost=prepost,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"Unexpected yfinance index format for {ticker_symbol}")

    df = df.reset_index()

    if isinstance(df.columns, pd.MultiIndex):
        flat_cols = []
        for col in df.columns:
            if isinstance(col, tuple):
                flat_cols.append(str(col[0]).lower())
            else:
                flat_cols.append(str(col).lower())
        df.columns = flat_cols
    else:
        df.columns = [str(c).lower() for c in df.columns]

    if "datetime" in df.columns:
        dt_col = "datetime"
    elif "date" in df.columns:
        dt_col = "date"
    else:
        raise ValueError(f"Unexpected yfinance dataframe format for {ticker_symbol}")

    df["datetime"] = pd.to_datetime(df[dt_col], utc=True).dt.tz_convert(NY_TZ)

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = 0.0

    out = df[["datetime", "open", "high", "low", "close", "volume"]].copy()

    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["datetime", "close"]).sort_values("datetime").reset_index(drop=True)
    return out


def get_intraday_history_last_24h_extended(ticker_symbol: str) -> pd.DataFrame:
    """
    Returns the last 24 hours worth of 1-minute bars (1440 bars max)
    from yfinance, including:
    - premarket
    - market hours
    - aftermarket
    - overnight

    This is bar-based instead of strict wall-clock-based so it still works
    on weekends and holidays.
    """
    df = _download_yf_intraday(
        ticker_symbol=ticker_symbol,
        period="5d",
        interval="1m",
        prepost=True,
    )

    if df.empty:
        raise ValueError(f"No intraday data found for {ticker_symbol}")

    # 24 hours of 1-minute bars
    df = df.tail(1440).copy()

    if df.empty:
        raise ValueError(f"No last-24h data found for {ticker_symbol}")

    df["session_date"] = df["datetime"].dt.date
    return df.reset_index(drop=True)


def get_current_spot_price(ticker_symbol: str) -> float:
    """
    Current displayed spot comes from the latest available 1-minute bar
    in the last 24 hours worth of bars.
    """
    df = get_intraday_history_last_24h_extended(ticker_symbol)
    if df.empty:
        raise ValueError(f"No current spot data for {ticker_symbol}")
    return float(df["close"].iloc[-1])


def get_latest_session_open_spot_price(ticker_symbol: str) -> float:
    """
    Uses yfinance to find the most recent regular-session 9:30 AM NY open.
    This is used to anchor OI.
    On weekends, this uses the last available trading session.
    """
    df = _download_yf_intraday(
        ticker_symbol=ticker_symbol,
        period="5d",
        interval="1m",
        prepost=False,
    )

    if df.empty:
        raise ValueError(f"No session open data for {ticker_symbol}")

    df = df[
        ((df["datetime"].dt.hour > 9) | ((df["datetime"].dt.hour == 9) & (df["datetime"].dt.minute >= 30)))
        & (df["datetime"].dt.hour < 16)
    ].copy()

    if df.empty:
        raise ValueError(f"No regular-session intraday data found for {ticker_symbol}")

    df["session_date"] = df["datetime"].dt.date
    latest_session = max(df["session_date"])
    day_df = df[df["session_date"] == latest_session].copy()

    open_bar = day_df[
        (day_df["datetime"].dt.hour == 9) & (day_df["datetime"].dt.minute == 30)
    ]

    if not open_bar.empty:
        return float(open_bar["open"].iloc[0])

    return float(day_df["open"].iloc[0])


def get_option_chain_snapshot_df(
    ticker_symbol: str,
    strike_price_gte: float | None = None,
    strike_price_lte: float | None = None,
) -> pd.DataFrame:
    """
    Options data still comes from Polygon/Massive.
    """
    client = get_client()

    params = {"limit": 250}
    if strike_price_gte is not None:
        params["strike_price.gte"] = strike_price_gte
    if strike_price_lte is not None:
        params["strike_price.lte"] = strike_price_lte

    rows = []
    for item in client.list_snapshot_options_chain(ticker_symbol, params=params):
        d = obj_to_dict(item)
        details = d.get("details", {}) or {}
        greeks = d.get("greeks", {}) or {}
        day = d.get("day", {}) or {}

        strike = details.get("strike_price")
        expiration_date = details.get("expiration_date")
        contract_type = details.get("contract_type")

        if strike is None or expiration_date is None or contract_type is None:
            continue

        rows.append(
            {
                "strike": float(strike),
                "expiration_date": str(expiration_date),
                "contract_type": str(contract_type).lower(),
                "open_interest": float(d.get("open_interest") or 0.0),
                "implied_volatility": float(d.get("implied_volatility") or 0.0),
                "gamma": float(greeks.get("gamma") or 0.0),
                "vega": float(greeks.get("vega") or 0.0),
                "day_volume": float(day.get("volume") or 0.0),
            }
        )

    if not rows:
        raise ValueError(f"No option chain snapshot rows for {ticker_symbol}")

    return pd.DataFrame(rows)


def get_first_n_expirations_from_chain_df(chain_df: pd.DataFrame, n: int) -> list[str]:
    expirations = sorted(chain_df["expiration_date"].dropna().astype(str).unique().tolist())
    if not expirations:
        raise ValueError("No expirations found in option chain snapshot.")
    return expirations[:n]


def get_weighted_option_data_polygon(
    ticker_symbol: str,
    weights: list[float],
    fixed_spot: float | None = None,
    max_distance: float | None = None,
):
    """
    Spot comes from yfinance.
    Options chain / greeks come from Polygon.
    """
    spot = float(fixed_spot) if fixed_spot is not None else get_current_spot_price(ticker_symbol)

    strike_gte = None
    strike_lte = None
    if max_distance is not None:
        strike_gte = max(0.01, spot - (max_distance * 1.5))
        strike_lte = spot + (max_distance * 1.5)

    chain_df = get_option_chain_snapshot_df(
        ticker_symbol=ticker_symbol,
        strike_price_gte=strike_gte,
        strike_price_lte=strike_lte,
    )

    expirations = get_first_n_expirations_from_chain_df(chain_df, len(weights))
    weight_map = {exp: w for exp, w in zip(expirations, weights)}

    chain_df = chain_df[chain_df["expiration_date"].isin(expirations)].copy()
    chain_df["weight"] = chain_df["expiration_date"].map(weight_map)
    chain_df["weighted_open_interest"] = chain_df["open_interest"] * chain_df["weight"]
    chain_df["weighted_volume"] = chain_df["day_volume"] * chain_df["weight"]

    chain_df["gex"] = 0.0
    chain_df.loc[chain_df["contract_type"] == "call", "gex"] = (
        chain_df["gamma"] * chain_df["open_interest"] * 100.0 * (spot ** 2) * 0.01
    )
    chain_df.loc[chain_df["contract_type"] == "put", "gex"] = (
        -chain_df["gamma"] * chain_df["open_interest"] * 100.0 * (spot ** 2) * 0.01
    )
    chain_df["weighted_gex"] = chain_df["gex"] * chain_df["weight"]

    chain_df["vex"] = chain_df["vega"] * chain_df["open_interest"] * 100.0
    chain_df.loc[chain_df["contract_type"] == "put", "vex"] = -chain_df.loc[
        chain_df["contract_type"] == "put", "vex"
    ]
    chain_df["weighted_vex"] = chain_df["vex"] * chain_df["weight"]

    calls = chain_df[chain_df["contract_type"] == "call"].copy()
    puts = chain_df[chain_df["contract_type"] == "put"].copy()

    combined_calls = (
        calls.groupby("strike", as_index=False)
        .agg(
            total_open_interest=("open_interest", "sum"),
            weighted_open_interest=("weighted_open_interest", "sum"),
            total_volume=("day_volume", "sum"),
            weighted_volume=("weighted_volume", "sum"),
            avg_implied_volatility=("implied_volatility", "mean"),
            total_gex=("gex", "sum"),
            weighted_gex=("weighted_gex", "sum"),
            total_vex=("vex", "sum"),
            weighted_vex=("weighted_vex", "sum"),
        )
        .sort_values("strike")
        .reset_index(drop=True)
    )

    combined_puts = (
        puts.groupby("strike", as_index=False)
        .agg(
            total_open_interest=("open_interest", "sum"),
            weighted_open_interest=("weighted_open_interest", "sum"),
            total_volume=("day_volume", "sum"),
            weighted_volume=("weighted_volume", "sum"),
            avg_implied_volatility=("implied_volatility", "mean"),
            total_gex=("gex", "sum"),
            weighted_gex=("weighted_gex", "sum"),
            total_vex=("vex", "sum"),
            weighted_vex=("weighted_vex", "sum"),
        )
        .sort_values("strike")
        .reset_index(drop=True)
    )

    return spot, expirations, combined_calls, combined_puts


def get_local_range(spot: float, max_distance: float):
    return spot - max_distance, spot + max_distance


def filter_local_calls(combined_calls: pd.DataFrame, spot: float, max_distance: float) -> pd.DataFrame:
    _, max_strike = get_local_range(spot, max_distance)
    return combined_calls[(combined_calls["strike"] > spot) & (combined_calls["strike"] <= max_strike)].copy()


def filter_local_puts(combined_puts: pd.DataFrame, spot: float, max_distance: float) -> pd.DataFrame:
    min_strike, _ = get_local_range(spot, max_distance)
    return combined_puts[(combined_puts["strike"] < spot) & (combined_puts["strike"] >= min_strike)].copy()


def choose_nearest_key_level(levels_df: pd.DataFrame, spot: float, score_column: str) -> float | None:
    if levels_df.empty:
        return None
    df = levels_df.copy()
    df["distance_to_spot"] = (df["strike"] - spot).abs()
    df = df.sort_values(by=["distance_to_spot", score_column], ascending=[True, False])
    return float(df.iloc[0]["strike"])