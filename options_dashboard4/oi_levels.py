import pandas as pd

from options_config import TICKER, EXPIRATION_WEIGHTS, MAX_DISTANCE, NUM_LEVELS
from options_common import (
    get_weighted_option_data_polygon,
    filter_local_calls,
    filter_local_puts,
    get_local_range,
)


def _empty_oi_df(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _oi_key_from_calls_puts(
    local_calls: pd.DataFrame,
    local_puts: pd.DataFrame,
    current_spot: float | None = None,
):
    """
    OI Key = strike with the strongest single-side weighted OI concentration.

    This intentionally does NOT use Net OI.

    At every visible local strike:
        dominant_weighted_oi = max(call_weighted_oi, put_weighted_oi)

    The key is the strike with the highest dominant_weighted_oi.
    """
    call_df = pd.DataFrame()
    put_df = pd.DataFrame()

    if local_calls is not None and not local_calls.empty:
        call_df = local_calls[["strike", "weighted_open_interest"]].copy()
        call_df = call_df.rename(columns={"weighted_open_interest": "call_weighted_oi"})

    if local_puts is not None and not local_puts.empty:
        put_df = local_puts[["strike", "weighted_open_interest"]].copy()
        put_df = put_df.rename(columns={"weighted_open_interest": "put_weighted_oi"})

    if call_df.empty and put_df.empty:
        return None

    if call_df.empty:
        call_df = pd.DataFrame(columns=["strike", "call_weighted_oi"])
    if put_df.empty:
        put_df = pd.DataFrame(columns=["strike", "put_weighted_oi"])

    merged = pd.merge(call_df, put_df, on="strike", how="outer").fillna(0.0)
    if merged.empty:
        return None

    merged["strike"] = pd.to_numeric(merged["strike"], errors="coerce")
    merged["call_weighted_oi"] = pd.to_numeric(merged["call_weighted_oi"], errors="coerce").fillna(0.0).abs()
    merged["put_weighted_oi"] = pd.to_numeric(merged["put_weighted_oi"], errors="coerce").fillna(0.0).abs()
    merged = merged.dropna(subset=["strike"])

    if merged.empty:
        return None

    merged["dominant_weighted_oi"] = merged[["call_weighted_oi", "put_weighted_oi"]].max(axis=1)
    merged["total_weighted_oi"] = merged["call_weighted_oi"] + merged["put_weighted_oi"]
    merged = merged[merged["dominant_weighted_oi"] > 0].copy()

    if merged.empty:
        return None

    if current_spot is not None:
        merged["distance_to_spot"] = (merged["strike"] - float(current_spot)).abs()
    else:
        merged["distance_to_spot"] = 0.0

    merged = merged.sort_values(
        by=["dominant_weighted_oi", "total_weighted_oi", "distance_to_spot", "strike"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    return float(merged.iloc[0]["strike"])


def _wall_from_df(df: pd.DataFrame, oi_column: str = "weighted_open_interest"):
    if df is None or df.empty or oi_column not in df.columns:
        return None

    clean = df[["strike", oi_column]].copy()
    clean["strike"] = pd.to_numeric(clean["strike"], errors="coerce")
    clean[oi_column] = pd.to_numeric(clean[oi_column], errors="coerce").fillna(0.0)
    clean = clean.dropna(subset=["strike"])
    clean = clean[clean[oi_column] > 0]

    if clean.empty:
        return None

    return float(clean.loc[clean[oi_column].idxmax(), "strike"])


def get_oi_levels(
    ticker_symbol: str = TICKER,
    weights=None,
    max_distance: float = MAX_DISTANCE,
    num_levels: int = NUM_LEVELS,
    fixed_spot: float | None = None,
):
    if weights is None:
        weights = EXPIRATION_WEIGHTS

    spot, expirations, combined_calls, combined_puts, strike_step = get_weighted_option_data_polygon(
        ticker_symbol=ticker_symbol,
        weights=weights,
        fixed_spot=fixed_spot,
        max_distance=max_distance,
    )

    local_calls = filter_local_calls(
        combined_calls,
        spot,
        max_distance,
        strike_step=strike_step,
    )
    local_puts = filter_local_puts(
        combined_puts,
        spot,
        max_distance,
        strike_step=strike_step,
    )

    top_resistances = (
        local_calls.sort_values("weighted_open_interest", ascending=False)
        .head(num_levels)
        .reset_index(drop=True)
    )
    top_supports = (
        local_puts.sort_values("weighted_open_interest", ascending=False)
        .head(num_levels)
        .reset_index(drop=True)
    )

    # Correct OI Key:
    # Use the strongest single-side weighted OI concentration from the visible local range.
    # This matches the OI chart logic and does not use Net OI.
    key_level = _oi_key_from_calls_puts(
        local_calls=local_calls,
        local_puts=local_puts,
        current_spot=float(spot),
    )

    search_min, search_max = get_local_range(spot, max_distance, strike_step=strike_step)

    call_wall = _wall_from_df(local_calls)
    put_wall = _wall_from_df(local_puts)

    output_cols = [
        "strike",
        "weighted_open_interest",
        "total_open_interest",
        "weighted_volume",
        "total_volume",
        "weighted_dex",
        "total_dex",
    ]

    def safe_cols(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return _empty_oi_df(output_cols)
        out = df.copy()
        for col in output_cols:
            if col not in out.columns:
                out[col] = 0.0
        return out[output_cols]

    return {
        "model": "OI",
        "ticker": ticker_symbol,
        "spot": round(float(spot), 2),
        "strike_step": float(strike_step),
        "expirations_used": expirations,
        "weights_used": weights,
        "search_range": [round(search_min, 2), round(search_max, 2)],
        "key_level": key_level,
        "call_wall": call_wall,
        "put_wall": put_wall,
        "top_resistances": safe_cols(top_resistances),
        "top_supports": safe_cols(top_supports),
    }
