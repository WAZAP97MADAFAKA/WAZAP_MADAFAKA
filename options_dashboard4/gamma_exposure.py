import pandas as pd

from options_config import TICKER, EXPIRATION_WEIGHTS, MAX_DISTANCE, NUM_LEVELS
from options_common import (
    get_weighted_option_data_polygon,
    filter_local_calls,
    filter_local_puts,
    filter_local_curve,
    get_local_search_range_from_filtered,
)


def estimate_gamma_flip(grouped_df: pd.DataFrame):
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
            weight = abs(prev_val) / (abs(prev_val) + abs(curr_val))
            return round(prev_strike + (curr_strike - prev_strike) * weight, 2)

    return None


def infer_gamma_regime_from_net_gex(spot: float, gamma_flip, total_net_gex: float):
    if gamma_flip is not None:
        return "ABOVE_FLIP_RANGE_BIAS" if spot > gamma_flip else "BELOW_FLIP_TREND_BIAS"

    if total_net_gex > 0:
        return "NO_LOCAL_FLIP_LONG_GAMMA_BIAS"
    if total_net_gex < 0:
        return "NO_LOCAL_FLIP_SHORT_GAMMA_BIAS"
    return "NO_NEARBY_FLIP_DETECTED"


def get_gamma_levels(
    ticker_symbol: str = TICKER,
    weights=None,
    max_distance: float = MAX_DISTANCE,
    num_levels: int = NUM_LEVELS,
    fixed_spot: float | None = None,
):
    if weights is None:
        weights = EXPIRATION_WEIGHTS

    live_spot, expirations_live, _, _, live_strike_step = get_weighted_option_data_polygon(
        ticker_symbol=ticker_symbol,
        weights=weights,
        fixed_spot=None,
        max_distance=max_distance,
    )

    anchor_spot = float(fixed_spot) if fixed_spot is not None else float(live_spot)

    _, expirations, combined_calls, combined_puts, strike_step = get_weighted_option_data_polygon(
        ticker_symbol=ticker_symbol,
        weights=weights,
        fixed_spot=anchor_spot,
        max_distance=max_distance,
    )

    local_calls = filter_local_calls(combined_calls, anchor_spot, max_distance, strike_step=strike_step)
    local_puts = filter_local_puts(combined_puts, anchor_spot, max_distance, strike_step=strike_step)

    top_resistances = (
        local_calls.sort_values("weighted_gex", ascending=False)
        .head(num_levels)
        .reset_index(drop=True)
    )

    top_supports = (
        local_puts.sort_values("weighted_gex", ascending=True)
        .head(num_levels)
        .reset_index(drop=True)
    )

    top_vex_resistances = (
        local_calls.sort_values("weighted_vex", ascending=False)
        .head(num_levels)
        .reset_index(drop=True)
    )

    top_vex_supports = (
        local_puts.sort_values("weighted_vex", ascending=True)
        .head(num_levels)
        .reset_index(drop=True)
    )

    _, _, wide_calls, wide_puts, wide_strike_step = get_weighted_option_data_polygon(
        ticker_symbol=ticker_symbol,
        weights=weights,
        fixed_spot=anchor_spot,
        max_distance=max_distance * 4,
    )

    combined_all = pd.concat(
        [
            wide_calls[["strike", "weighted_gex"]].copy(),
            wide_puts[["strike", "weighted_gex"]].copy(),
        ],
        ignore_index=True,
    )

    combined_all = (
        combined_all.groupby("strike", as_index=False)
        .agg(weighted_gex=("weighted_gex", "sum"))
        .sort_values("strike")
        .reset_index(drop=True)
    )

    gamma_flip = estimate_gamma_flip(combined_all)
    total_net_gex = float(combined_all["weighted_gex"].sum()) if not combined_all.empty else 0.0

    search_min, search_max = get_local_range(anchor_spot, max_distance, strike_step=strike_step)

    gamma_key_global = None
    if not combined_all.empty:
        combined_all["abs_weighted_gex"] = combined_all["weighted_gex"].abs()
        global_curve = combined_all.dropna(subset=["strike", "abs_weighted_gex"]).copy()

        if not global_curve.empty:
            gamma_key_global = float(
                global_curve.loc[global_curve["abs_weighted_gex"].idxmax(), "strike"]
            )

    gamma_key_local = None
    local_curve_for_key = combined_all[
        (combined_all["strike"] >= search_min) & (combined_all["strike"] <= search_max)
    ].copy()

    if not local_curve_for_key.empty:
        if "abs_weighted_gex" not in local_curve_for_key.columns:
            local_curve_for_key["abs_weighted_gex"] = local_curve_for_key["weighted_gex"].abs()

        local_curve_for_key = local_curve_for_key.dropna(subset=["strike", "abs_weighted_gex"])

        if not local_curve_for_key.empty:
            gamma_key_local = float(
                local_curve_for_key.loc[
                    local_curve_for_key["abs_weighted_gex"].idxmax(), "strike"
                ]
            )

    key_level = gamma_key_local if gamma_key_local is not None else gamma_key_global

    regime = infer_gamma_regime_from_net_gex(float(live_spot), gamma_flip, total_net_gex)

    local_gex_curve = combined_all[
        (combined_all["strike"] >= search_min) & (combined_all["strike"] <= search_max)
    ].copy()

    return {
        "model": "GAMMA",
        "ticker": ticker_symbol,
        "spot": round(float(live_spot), 2),
        "anchor_spot": round(float(anchor_spot), 2),
        "strike_step": strike_step,
        "expirations_used": expirations,
        "weights_used": weights,
        "search_range": [round(search_min, 2), round(search_max, 2)],
        "gamma_flip": gamma_flip,
        "key_level": key_level,
        "gamma_key_global": gamma_key_global,
        "gamma_key_local": gamma_key_local,
        "regime": regime,
        "total_net_gex": total_net_gex,
        "flip_source": "wide_scan" if gamma_flip is not None else "net_gex_proxy",
        "top_resistances": top_resistances[
            ["strike", "weighted_gex", "total_gex", "weighted_vex", "total_vex", "total_open_interest", "weighted_volume"]
        ],
        "top_supports": top_supports[
            ["strike", "weighted_gex", "total_gex", "weighted_vex", "total_vex", "total_open_interest", "weighted_volume"]
        ],
        "top_vex_resistances": top_vex_resistances[
            ["strike", "weighted_vex", "total_vex", "weighted_gex", "total_gex"]
        ],
        "top_vex_supports": top_vex_supports[
            ["strike", "weighted_vex", "total_vex", "weighted_gex", "total_gex"]
        ],
        "gex_curve": local_gex_curve.to_dict(orient="records"),
        "gex_curve_wide": combined_all.to_dict(orient="records"),
    }
