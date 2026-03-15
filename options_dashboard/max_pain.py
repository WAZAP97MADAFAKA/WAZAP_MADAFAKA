import pandas as pd

from options_config import TICKER, EXPIRATION_WEIGHTS, MAX_DISTANCE, NUM_LEVELS
from options_common import (
    get_weighted_option_data,
    get_local_range,
)


def get_max_pain_levels(
    ticker_symbol: str = TICKER,
    weights=None,
    max_distance: float = MAX_DISTANCE,
    num_levels: int = NUM_LEVELS
):
    if weights is None:
        weights = EXPIRATION_WEIGHTS

    spot, expirations, combined_calls, combined_puts = get_weighted_option_data(ticker_symbol, weights)
    search_min, search_max = get_local_range(spot, max_distance)

    strikes = sorted(set(combined_calls["strike"]).intersection(set(combined_puts["strike"])))
    local_strikes = [s for s in strikes if search_min <= s <= search_max]

    pain_rows = []

    for settlement_price in local_strikes:
        call_pain = (
            ((settlement_price - combined_calls["strike"]).clip(lower=0))
            * combined_calls["weighted_open_interest"]
            * 100
        ).sum()

        put_pain = (
            ((combined_puts["strike"] - settlement_price).clip(lower=0))
            * combined_puts["weighted_open_interest"]
            * 100
        ).sum()

        total_pain = call_pain + put_pain

        pain_rows.append({
            "strike": float(settlement_price),
            "call_pain": float(call_pain),
            "put_pain": float(put_pain),
            "total_pain": float(total_pain),
        })

    pain_df = pd.DataFrame(pain_rows).sort_values("strike").reset_index(drop=True)

    if pain_df.empty:
        return {
            "model": "MAX_PAIN",
            "ticker": ticker_symbol,
            "spot": round(spot, 2),
            "expirations_used": expirations,
            "weights_used": weights,
            "search_range": [round(search_min, 2), round(search_max, 2)],
            "key_level": None,
            "top_resistances": pd.DataFrame(),
            "top_supports": pd.DataFrame(),
        }

    key_level = float(pain_df.sort_values("total_pain", ascending=True).iloc[0]["strike"])

    above = pain_df[pain_df["strike"] > spot].copy()
    below = pain_df[pain_df["strike"] < spot].copy()

    top_resistances = (
        above.sort_values("total_pain", ascending=True)
        .head(num_levels)
        .sort_values("strike")
        .reset_index(drop=True)
    )

    top_supports = (
        below.sort_values("total_pain", ascending=True)
        .head(num_levels)
        .sort_values("strike", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "model": "MAX_PAIN",
        "ticker": ticker_symbol,
        "spot": round(spot, 2),
        "expirations_used": expirations,
        "weights_used": weights,
        "search_range": [round(search_min, 2), round(search_max, 2)],
        "key_level": key_level,
        "top_resistances": top_resistances[[
            "strike", "total_pain", "call_pain", "put_pain"
        ]],
        "top_supports": top_supports[[
            "strike", "total_pain", "call_pain", "put_pain"
        ]],
    }


if __name__ == "__main__":
    result = get_max_pain_levels()

    print("\nLOCAL MAX PAIN LEVELS")
    print({
        "ticker": result["ticker"],
        "spot": result["spot"],
        "expirations_used": result["expirations_used"],
        "weights_used": result["weights_used"],
        "search_range": result["search_range"],
        "key_level": result["key_level"],
    })

    print("\nTop Resistances:")
    print(result["top_resistances"].to_string(index=False))

    print("\nTop Supports:")
    print(result["top_supports"].to_string(index=False))