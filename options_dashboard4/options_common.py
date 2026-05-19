import os
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from massive import RESTClient

from options_config import NY_TIMEZONE

NY_TZ = ZoneInfo(NY_TIMEZONE)


def get_price_ticker_symbol(ticker_symbol: str) -> str:
    mapping = {
        "SPX": "^SPX",
        "NDX": "^NDX",
    }
    return mapping.get(str(ticker_symbol).upper(), ticker_symbol)


def get_options_ticker_symbol(ticker_symbol: str) -> str:
    mapping = {
        "SPX": "I:SPX",
        "NDX": "I:NDX",
    }
    return mapping.get(str(ticker_symbol).upper(), ticker_symbol)


def get_default_strike_step(ticker_symbol: str) -> float:
    ticker_symbol = str(ticker_symbol).upper()
    if ticker_symbol == "SPX":
        return 5.0
    if ticker_symbol == "NDX":
        return 10.0
    return 1.0


def normalize_strike_step(raw_step: float, ticker_symbol: str) -> float:
    raw_step = float(raw_step or 0)
    fallback = get_default_strike_step(ticker_symbol)
    if raw_step <= 0:
        return fallback
    if raw_step <= 1:
        return 1.0
    if raw_step <= 2.5:
        return 2.5
    if raw_step <= 5:
        return 5.0
    if raw_step <= 10:
        return 10.0
    if raw_step <= 25:
        return 25.0
    if raw_step <= 50:
        return 50.0
    return 100.0


def infer_strike_step(chain_df: pd.DataFrame, ticker_symbol: str) -> float:
    fallback = get_default_strike_step(ticker_symbol)
    if chain_df is None or chain_df.empty or "strike" not in chain_df.columns:
        return fallback

    strikes = sorted(pd.to_numeric(chain_df["strike"], errors="coerce").dropna().unique().tolist())
    if len(strikes) < 2:
        return fallback

    diffs = []
    for i in range(1, len(strikes)):
        diff = round(float(strikes[i]) - float(strikes[i - 1]), 6)
        if diff > 0:
            diffs.append(diff)

    if not diffs:
        return fallback

    mode_vals = pd.Series(diffs).mode()
    raw_step = float(mode_vals.iloc[0]) if not mode_vals.empty else float(min(diffs))
    return normalize_strike_step(raw_step, ticker_symbol)



def _safe_float(value):
    try:
        if value is None:
            return None
        value = float(value)
        if pd.isna(value) or value <= 0:
            return None
        return value
    except Exception:
        return None


def extract_massive_underlying_price(snapshot_dict: dict):
    """
    Extract Massive's underlying asset price from an option-chain snapshot row.

    Massive's normal field is expected to be:
        underlying_asset.price

    The fallback paths below make the dashboard more robust if the client object
    shape changes slightly.
    """
    if not isinstance(snapshot_dict, dict):
        return None

    underlying = snapshot_dict.get("underlying_asset", {}) or {}
    candidate_values = []

    if isinstance(underlying, dict):
        candidate_values.extend(
            [
                underlying.get("price"),
                underlying.get("last_price"),
                underlying.get("close"),
                underlying.get("value"),
            ]
        )

        day = underlying.get("day", {}) or {}
        last_trade = underlying.get("last_trade", {}) or {}
        last_quote = underlying.get("last_quote", {}) or {}

        if isinstance(day, dict):
            candidate_values.extend([day.get("close"), day.get("c")])

        if isinstance(last_trade, dict):
            candidate_values.extend([last_trade.get("price"), last_trade.get("p")])

        if isinstance(last_quote, dict):
            bid = _safe_float(last_quote.get("bid") or last_quote.get("bid_price"))
            ask = _safe_float(last_quote.get("ask") or last_quote.get("ask_price"))
            if bid is not None and ask is not None:
                candidate_values.append((bid + ask) / 2.0)

    for value in candidate_values:
        parsed = _safe_float(value)
        if parsed is not None:
            return parsed

    return None


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
    yf_ticker = get_price_ticker_symbol(ticker_symbol)

    df = yf.download(
        tickers=yf_ticker,
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
        df.columns = [str(col[0]).lower() if isinstance(col, tuple) else str(col).lower() for col in df.columns]
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

    return out.dropna(subset=["datetime", "close"]).sort_values("datetime").reset_index(drop=True)


def get_intraday_history_last_24h_extended(ticker_symbol: str) -> pd.DataFrame:
    df = _download_yf_intraday(
        ticker_symbol=ticker_symbol,
        period="5d",
        interval="1m",
        prepost=True,
    )
    if df.empty:
        raise ValueError(f"No intraday data found for {ticker_symbol}")
    df = df.tail(1440).copy()
    if df.empty:
        raise ValueError(f"No last-24h data found for {ticker_symbol}")
    df["session_date"] = df["datetime"].dt.date
    return df.reset_index(drop=True)


def get_current_spot_price(ticker_symbol: str) -> float:
    df = get_intraday_history_last_24h_extended(ticker_symbol)
    if df.empty:
        raise ValueError(f"No current spot data for {ticker_symbol}")
    return float(df["close"].iloc[-1])


def get_latest_session_open_spot_price(ticker_symbol: str) -> float:
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

    open_bar = day_df[(day_df["datetime"].dt.hour == 9) & (day_df["datetime"].dt.minute == 30)]
    if not open_bar.empty:
        return float(open_bar["open"].iloc[0])
    return float(day_df["open"].iloc[0])


def get_option_chain_snapshot_df(
    ticker_symbol: str,
    strike_price_gte: float | None = None,
    strike_price_lte: float | None = None,
) -> pd.DataFrame:
    client = get_client()
    options_ticker = get_options_ticker_symbol(ticker_symbol)

    params = {"limit": 250}
    if strike_price_gte is not None:
        params["strike_price.gte"] = strike_price_gte
    if strike_price_lte is not None:
        params["strike_price.lte"] = strike_price_lte

    rows = []
    for item in client.list_snapshot_options_chain(options_ticker, params=params):
        d = obj_to_dict(item)
        details = d.get("details", {}) or {}
        greeks = d.get("greeks", {}) or {}
        day = d.get("day", {}) or {}
        underlying_price = extract_massive_underlying_price(d)

        strike = details.get("strike_price")
        expiration_date = details.get("expiration_date")
        contract_type = details.get("contract_type")

        if strike is None or expiration_date is None or contract_type is None:
            continue

        contract_type = str(contract_type).lower()
        delta = float(greeks.get("delta") or 0.0)
        if contract_type == "put" and delta > 0:
            delta = -delta
        if contract_type == "call" and delta < 0:
            delta = abs(delta)

        rows.append(
            {
                "strike": float(strike),
                "expiration_date": str(expiration_date),
                "contract_type": contract_type,
                "open_interest": float(d.get("open_interest") or 0.0),
                "implied_volatility": float(d.get("implied_volatility") or 0.0),
                "gamma": float(greeks.get("gamma") or 0.0),
                "vega": float(greeks.get("vega") or 0.0),
                "delta": delta,
                "underlying_price": underlying_price,
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
    dex_spot: float | None = None,
    return_metadata: bool = False,
):
    """
    Fetches ONE option-chain snapshot for the requested ticker/range and builds
    all OI/GEX/VEX/DEX aggregates from that same snapshot.

    Important spots:
    - query_spot: used only to choose the option-chain strike range.
    - exposure_spot: used for GEX dollars and Notional DEX dollars.
      Preferred source is Massive's underlying_asset.price from the snapshot.
    - returned_spot: kept backward compatible. If fixed_spot is passed, this
      returns the anchor spot so old OI/range logic does not break.

    return_metadata=True adds:
        query_spot, exposure_spot, massive_underlying_spot, returned_spot
    without changing old callers that expect the original 5-item tuple.
    """
    query_spot = float(fixed_spot) if fixed_spot is not None else get_current_spot_price(ticker_symbol)
    provisional_step = get_default_strike_step(ticker_symbol)

    strike_gte = None
    strike_lte = None
    if max_distance is not None:
        query_distance_points = float(max_distance) * provisional_step * 1.5
        strike_gte = max(0.01, query_spot - query_distance_points)
        strike_lte = query_spot + query_distance_points

    # This is the only Massive option-chain snapshot call made by this function.
    chain_df = get_option_chain_snapshot_df(
        ticker_symbol=ticker_symbol,
        strike_price_gte=strike_gte,
        strike_price_lte=strike_lte,
    )

    massive_underlying_spot = None
    if "underlying_price" in chain_df.columns:
        underlying_prices = pd.to_numeric(chain_df["underlying_price"], errors="coerce").dropna()
        underlying_prices = underlying_prices[underlying_prices > 0]
        if not underlying_prices.empty:
            massive_underlying_spot = float(underlying_prices.iloc[0])

    if massive_underlying_spot is not None:
        exposure_spot = float(massive_underlying_spot)
    elif dex_spot is not None:
        exposure_spot = float(dex_spot)
    else:
        exposure_spot = float(query_spot)

    returned_spot = (
        float(massive_underlying_spot)
        if fixed_spot is None and massive_underlying_spot is not None
        else float(query_spot)
    )

    strike_step = infer_strike_step(chain_df, ticker_symbol)
    expirations = get_first_n_expirations_from_chain_df(chain_df, len(weights))
    weight_map = {exp: w for exp, w in zip(expirations, weights)}

    chain_df = chain_df[chain_df["expiration_date"].isin(expirations)].copy()
    chain_df["weight"] = chain_df["expiration_date"].map(weight_map).fillna(0.0)
    chain_df["weighted_open_interest"] = chain_df["open_interest"] * chain_df["weight"]
    chain_df["weighted_volume"] = chain_df["day_volume"] * chain_df["weight"]

    # GEX dollars. Uses Massive underlying price when available.
    chain_df["gex"] = 0.0
    chain_df.loc[chain_df["contract_type"] == "call", "gex"] = (
        chain_df["gamma"] * chain_df["open_interest"] * 100.0 * (exposure_spot ** 2) * 0.01
    )
    chain_df.loc[chain_df["contract_type"] == "put", "gex"] = (
        -chain_df["gamma"] * chain_df["open_interest"] * 100.0 * (exposure_spot ** 2) * 0.01
    )
    chain_df["weighted_gex"] = chain_df["gex"] * chain_df["weight"]

    # VEX does not use spot.
    chain_df["vex"] = chain_df["vega"] * chain_df["open_interest"] * 100.0
    chain_df.loc[chain_df["contract_type"] == "put", "vex"] = -chain_df.loc[
        chain_df["contract_type"] == "put", "vex"
    ]
    chain_df["weighted_vex"] = chain_df["vex"] * chain_df["weight"]

    # Notional DEX dollars. Uses Massive underlying price when available.
    chain_df["dex"] = chain_df["delta"] * chain_df["open_interest"] * 100.0 * exposure_spot
    chain_df["weighted_dex"] = chain_df["dex"] * chain_df["weight"]

    calls = chain_df[chain_df["contract_type"] == "call"].copy()
    puts = chain_df[chain_df["contract_type"] == "put"].copy()

    agg_map = dict(
        total_open_interest=("open_interest", "sum"),
        weighted_open_interest=("weighted_open_interest", "sum"),
        total_volume=("day_volume", "sum"),
        weighted_volume=("weighted_volume", "sum"),
        avg_implied_volatility=("implied_volatility", "mean"),
        avg_delta=("delta", "mean"),
        total_gex=("gex", "sum"),
        weighted_gex=("weighted_gex", "sum"),
        total_vex=("vex", "sum"),
        weighted_vex=("weighted_vex", "sum"),
        total_dex=("dex", "sum"),
        weighted_dex=("weighted_dex", "sum"),
    )

    combined_calls = (
        calls.groupby("strike", as_index=False)
        .agg(**agg_map)
        .sort_values("strike")
        .reset_index(drop=True)
    )
    combined_puts = (
        puts.groupby("strike", as_index=False)
        .agg(**agg_map)
        .sort_values("strike")
        .reset_index(drop=True)
    )

    metadata = {
        "query_spot": float(query_spot),
        "returned_spot": float(returned_spot),
        "massive_underlying_spot": massive_underlying_spot,
        "exposure_spot": float(exposure_spot),
        "strike_gte": strike_gte,
        "strike_lte": strike_lte,
    }

    if return_metadata:
        return returned_spot, expirations, combined_calls, combined_puts, float(strike_step), metadata

    return returned_spot, expirations, combined_calls, combined_puts, float(strike_step)


def get_local_range(spot: float, max_distance: float, strike_step: float = 1.0):
    distance_points = float(max_distance) * float(strike_step or 1.0)
    return spot - distance_points, spot + distance_points


def filter_local_calls(
    combined_calls: pd.DataFrame,
    spot: float,
    max_distance: float,
    strike_step: float = 1.0,
) -> pd.DataFrame:
    _, max_strike = get_local_range(spot, max_distance, strike_step)
    return combined_calls[(combined_calls["strike"] > spot) & (combined_calls["strike"] <= max_strike)].copy()


def filter_local_puts(
    combined_puts: pd.DataFrame,
    spot: float,
    max_distance: float,
    strike_step: float = 1.0,
) -> pd.DataFrame:
    min_strike, _ = get_local_range(spot, max_distance, strike_step)
    return combined_puts[(combined_puts["strike"] < spot) & (combined_puts["strike"] >= min_strike)].copy()


def choose_nearest_key_level(levels_df: pd.DataFrame, spot: float, score_column: str) -> float | None:
    if levels_df.empty:
        return None
    df = levels_df.copy()
    df["distance_to_spot"] = (df["strike"] - spot).abs()
    df = df.sort_values(by=["distance_to_spot", score_column], ascending=[True, False])
    return float(df.iloc[0]["strike"])