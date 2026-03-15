import pandas as pd
import yfinance as yf

from options_config import TICKER, EXPIRATION_WEIGHTS, MAX_DISTANCE, NUM_LEVELS
from options_common import (
    get_spot_price,
    get_first_n_expirations,
    filter_local_calls,
    filter_local_puts,
    choose_nearest_key_level,
    get_local_range,
    time_to_expiration_in_years,
    bs_gamma,
)


def estimate_gamma_flip_from_strikes(grouped_df: pd.DataFrame):
    """
    Approximate gamma flip by finding where cumulative weighted GEX
    changes sign across strikes.
    """
    if grouped_df.empty:
        return None

    df = grouped_df.copy().sort_values("strike").reset_index(drop=True)
    df["cum_weighted_gex"] = df["weighted_gex"].cumsum()

    for i in range(1, len(df)):
        prev_val = df.loc[i - 1, "cum_weighted_gex"]
        curr_val = df.loc[i, "cum_weighted_gex"]

        if prev_val == 0:
            return float(df.loc[i - 1, "strike"])

        if prev_val * curr_val < 0:
            prev_strike = float(df.loc[i - 1, "strike"])
            curr_strike = float(df.loc[i, "strike"])

            # Linear interpolation between the two strikes
            weight = abs(prev_val) / (abs(prev_val) + abs(curr_val))
            gamma_flip = prev_strike + (curr_strike - prev_strike) * weight
            return float(round(gamma_flip, 2))

    return None


def get_gamma_levels(
    ticker_symbol: str = TICKER,
    weights=None,
    max_distance: float = MAX_DISTANCE,
    num_levels: int = NUM_LEVELS
):
    if weights is None:
        weights = EXPIRATION_WEIGHTS

    ticker = yf.Ticker(ticker_symbol)
    spot = get_spot_price(ticker_symbol)
    expirations = get_first_n_expirations(ticker_symbol, len(weights))

    all_calls = []
    all_puts = []

    for exp, weight in zip(expirations, weights):
        chain = ticker.option_chain(exp)
        T = time_to_expiration_in_years(exp)

        calls = chain.calls.copy()
        puts = chain.puts.copy()

        for df in (calls, puts):
            df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
            df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce").fillna(0)
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
            df["impliedVolatility"] = pd.to_numeric(df["impliedVolatility"], errors="coerce").fillna(0)

        calls["gamma"] = calls.apply(
            lambda row: bs_gamma(spot, float(row["strike"]), T, float(row["impliedVolatility"])),
            axis=1
        )
        puts["gamma"] = puts.apply(
            lambda row: bs_gamma(spot, float(row["strike"]), T, float(row["impliedVolatility"])),
            axis=1
        )

        # Simplified GEX:
        # calls positive, puts negative
        calls["gex"] = calls["gamma"] * calls["openInterest"] * 100 * (spot ** 2) * 0.01
        puts["gex"] = -puts["gamma"] * puts["openInterest"] * 100 * (spot ** 2) * 0.01

        calls["weighted_gex"] = calls["gex"] * weight
        puts["weighted_gex"] = puts["gex"] * weight

        calls["weighted_volume"] = calls["volume"] * weight
        puts["weighted_volume"] = puts["volume"] * weight

        all_calls.append(calls[[
            "strike", "openInterest", "volume", "weighted_volume",
            "gex", "weighted_gex"
        ]])

        all_puts.append(puts[[
            "strike", "openInterest", "volume", "weighted_volume",
            "gex", "weighted_gex"
        ]])

    combined_calls = (
        pd.concat(all_calls, ignore_index=True)
        .groupby("strike", as_index=False)
        .agg(
            total_open_interest=("openInterest", "sum"),
            total_volume=("volume", "sum"),
            weighted_volume=("weighted_volume", "sum"),
            total_gex=("gex", "sum"),
            weighted_gex=("weighted_gex", "sum"),
        )
        .sort_values("strike")
        .reset_index(drop=True)
    )

    combined_puts = (
        pd.concat(all_puts, ignore_index=True)
        .groupby("strike", as_index=False)
        .agg(
            total_open_interest=("openInterest", "sum"),
            total_volume=("volume", "sum"),
            weighted_volume=("weighted_volume", "sum"),
            total_gex=("gex", "sum"),
            weighted_gex=("weighted_gex", "sum"),
        )
        .sort_values("strike")
        .reset_index(drop=True)
    )

    # Full gamma profile for gamma flip
    combined_all = pd.concat([
        combined_calls[["strike", "weighted_gex"]].copy(),
        combined_puts[["strike", "weighted_gex"]].copy()
    ], ignore_index=True)

    combined_all = (
        combined_all.groupby("strike", as_index=False)
        .agg(weighted_gex=("weighted_gex", "sum"))
        .sort_values("strike")
        .reset_index(drop=True)
    )

    gamma_flip = estimate_gamma_flip_from_strikes(combined_all)

    local_calls = filter_local_calls(combined_calls, spot, max_distance)
    local_puts = filter_local_puts(combined_puts, spot, max_distance)

    top_resistances = (
        local_calls.sort_values("weighted_gex", ascending=False)
        .head(num_levels)
        .sort_values("strike")
        .reset_index(drop=True)
    )

    # put GEX is negative, so strongest supports are most negative
    top_supports = (
        local_puts.sort_values("weighted_gex", ascending=True)
        .head(num_levels)
        .sort_values("strike", ascending=False)
        .reset_index(drop=True)
    )

    local_all = pd.concat([
        local_calls.assign(side="call"),
        local_puts.assign(side="put")
    ], ignore_index=True)

    local_all["abs_weighted_gex"] = local_all["weighted_gex"].abs()
    key_level = choose_nearest_key_level(local_all, spot, "abs_weighted_gex")

    search_min, search_max = get_local_range(spot, max_distance)

    return {
        "model": "GAMMA",
        "ticker": ticker_symbol,
        "spot": round(spot, 2),
        "expirations_used": expirations,
        "weights_used": weights,
        "search_range": [round(search_min, 2), round(search_max, 2)],
        "gamma_flip": gamma_flip,
        "key_level": key_level,
        "top_resistances": top_resistances[[
            "strike", "weighted_gex", "total_gex",
            "total_open_interest", "weighted_volume", "total_volume"
        ]],
        "top_supports": top_supports[[
            "strike", "weighted_gex", "total_gex",
            "total_open_interest", "weighted_volume", "total_volume"
        ]],
    }


if __name__ == "__main__":
    result = get_gamma_levels()

    print("\nLOCAL GAMMA LEVELS")
    print({
        "ticker": result["ticker"],
        "spot": result["spot"],
        "expirations_used": result["expirations_used"],
        "weights_used": result["weights_used"],
        "search_range": result["search_range"],
        "gamma_flip": result["gamma_flip"],
        "key_level": result["key_level"],
    })

    print("\nTop Resistances:")
    print(
        result["top_resistances"]
        .sort_values(by="weighted_gex",
        ascending=False)
        .head(NUM_LEVELS)
        .to_string(index=False))


    print("\nTop Supports:")
    print(
        result["top_supports"]
        .sort_values(by="weighted_gex",
        ascending=False)
        .head(NUM_LEVELS)
        .to_string(index=False))