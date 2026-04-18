import pandas as pd

from options_config import TICKER, EXPIRATION_WEIGHTS, MAX_DISTANCE, NUM_LEVELS
from options_common import (
    get_weighted_option_data_polygon,
    filter_local_calls,
    filter_local_puts,
    choose_nearest_key_level,
    get_local_range,
)


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

    combined_levels = pd.concat(
        [
            local_calls[["strike", "weighted_open_interest"]].copy(),
            local_puts[["strike", "weighted_open_interest"]].copy(),
        ],
        ignore_index=True,
    )

    # 🔥 TRUE OI KEY (dominant level, NOT nearest to spot)
    combined_levels = combined_levels.dropna(subset=["strike", "weighted_open_interest"])

    if not combined_levels.empty:
        combined_levels = combined_levels.sort_values(
            "weighted_open_interest", ascending=False
        ).reset_index(drop=True)
        key_level = float(combined_levels.iloc[0]["strike"])
    else:
        key_level = None

    search_min, search_max = get_local_range(spot, max_distance, strike_step=strike_step)

    return {
        "model": "OI",
        "ticker": ticker_symbol,
        "spot": round(spot, 2),
        "strike_step": float(strike_step),
        "expirations_used": expirations,
        "weights_used": weights,
        "search_range": [round(search_min, 2), round(search_max, 2)],
        "key_level": key_level,
        "top_resistances": top_resistances[
            ["strike", "weighted_open_interest", "total_open_interest", "weighted_volume", "total_volume"]
        ],
        "top_supports": top_supports[
            ["strike", "weighted_open_interest", "total_open_interest", "weighted_volume", "total_volume"]
        ],
    }
