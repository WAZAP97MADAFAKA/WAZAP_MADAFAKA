import pandas as pd

from options_config import TICKER, EXPIRATION_WEIGHTS, MAX_DISTANCE, NUM_LEVELS
from options_common import (
    get_weighted_option_data_polygon,
    filter_local_calls,
    filter_local_puts,
    get_local_range,
)


def _wall_from_df(df: pd.DataFrame, oi_column: str = "weighted_open_interest"):
    if df is None or df.empty or oi_column not in df.columns:
        return None
    clean = df.dropna(subset=["strike", oi_column]).copy()
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

    local_calls = filter_local_calls(combined_calls, spot, max_distance, strike_step=strike_step)
    local_puts = filter_local_puts(combined_puts, spot, max_distance, strike_step=strike_step)

    top_resistances = local_calls.sort_values("weighted_open_interest", ascending=False).head(num_levels).reset_index(drop=True)
    top_supports = local_puts.sort_values("weighted_open_interest", ascending=False).head(num_levels).reset_index(drop=True)

    combined_levels = pd.concat(
        [
            local_calls[["strike", "weighted_open_interest"]].copy(),
            local_puts[["strike", "weighted_open_interest"]].copy(),
        ],
        ignore_index=True,
    ).dropna(subset=["strike", "weighted_open_interest"])

    if not combined_levels.empty:
        combined_levels = combined_levels.sort_values("weighted_open_interest", ascending=False).reset_index(drop=True)
        key_level = float(combined_levels.iloc[0]["strike"])
    else:
        key_level = None

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

    def safe_cols(df):
        if df is None or df.empty:
            return pd.DataFrame(columns=output_cols)
        for c in output_cols:
            if c not in df.columns:
                df[c] = 0.0
        return df[output_cols]

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
