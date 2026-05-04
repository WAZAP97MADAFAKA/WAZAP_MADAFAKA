import pandas as pd

from options_config import TICKER, EXPIRATION_WEIGHTS, MAX_DISTANCE, NUM_LEVELS
from options_common import (
    get_weighted_option_data_polygon,
    filter_local_calls,
    filter_local_puts,
    get_local_range,
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


def _empty_curve_df():
    return pd.DataFrame(
        columns=[
            "strike",
            "weighted_gex",
            "total_gex",
            "weighted_vex",
            "total_vex",
            "weighted_dex",
            "total_dex",
            "weighted_open_interest",
            "total_open_interest",
            "call_weighted_oi",
            "put_weighted_oi",
            "call_weighted_dex",
            "put_weighted_dex",
        ]
    )


def build_combined_curve(calls: pd.DataFrame, puts: pd.DataFrame) -> pd.DataFrame:
    call_cols = [
        "strike",
        "weighted_gex",
        "total_gex",
        "weighted_vex",
        "total_vex",
        "weighted_dex",
        "total_dex",
        "weighted_open_interest",
        "total_open_interest",
    ]
    put_cols = call_cols.copy()

    call_df = calls[call_cols].copy() if calls is not None and not calls.empty else pd.DataFrame(columns=call_cols)
    put_df = puts[put_cols].copy() if puts is not None and not puts.empty else pd.DataFrame(columns=put_cols)

    for col in call_cols:
        if col not in call_df.columns:
            call_df[col] = 0.0
        if col not in put_df.columns:
            put_df[col] = 0.0

    call_df = call_df.rename(
        columns={
            "weighted_gex": "call_weighted_gex",
            "total_gex": "call_total_gex",
            "weighted_vex": "call_weighted_vex",
            "total_vex": "call_total_vex",
            "weighted_dex": "call_weighted_dex",
            "total_dex": "call_total_dex",
            "weighted_open_interest": "call_weighted_oi",
            "total_open_interest": "call_total_oi",
        }
    )
    put_df = put_df.rename(
        columns={
            "weighted_gex": "put_weighted_gex",
            "total_gex": "put_total_gex",
            "weighted_vex": "put_weighted_vex",
            "total_vex": "put_total_vex",
            "weighted_dex": "put_weighted_dex",
            "total_dex": "put_total_dex",
            "weighted_open_interest": "put_weighted_oi",
            "total_open_interest": "put_total_oi",
        }
    )

    merged = pd.merge(call_df, put_df, on="strike", how="outer").fillna(0.0)
    if merged.empty:
        return _empty_curve_df()

    merged["weighted_gex"] = merged["call_weighted_gex"] + merged["put_weighted_gex"]
    merged["total_gex"] = merged["call_total_gex"] + merged["put_total_gex"]
    merged["weighted_vex"] = merged["call_weighted_vex"] + merged["put_weighted_vex"]
    merged["total_vex"] = merged["call_total_vex"] + merged["put_total_vex"]
    merged["weighted_dex"] = merged["call_weighted_dex"] + merged["put_weighted_dex"]
    merged["total_dex"] = merged["call_total_dex"] + merged["put_total_dex"]
    merged["weighted_open_interest"] = merged["call_weighted_oi"] + merged["put_weighted_oi"]
    merged["total_open_interest"] = merged["call_total_oi"] + merged["put_total_oi"]

    out_cols = [
        "strike",
        "weighted_gex",
        "total_gex",
        "weighted_vex",
        "total_vex",
        "weighted_dex",
        "total_dex",
        "weighted_open_interest",
        "total_open_interest",
        "call_weighted_oi",
        "put_weighted_oi",
        "call_weighted_dex",
        "put_weighted_dex",
    ]
    return merged[out_cols].sort_values("strike").reset_index(drop=True)


def _wall_from_df(df: pd.DataFrame, oi_column: str = "weighted_open_interest"):
    if df is None or df.empty or oi_column not in df.columns:
        return None
    clean = df.dropna(subset=["strike", oi_column]).copy()
    if clean.empty:
        return None
    return float(clean.loc[clean[oi_column].idxmax(), "strike"])


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

    top_resistances = local_calls.sort_values("weighted_gex", ascending=False).head(num_levels).reset_index(drop=True)
    top_supports = local_puts.sort_values("weighted_gex", ascending=True).head(num_levels).reset_index(drop=True)
    top_vex_resistances = local_calls.sort_values("weighted_vex", ascending=False).head(num_levels).reset_index(drop=True)
    top_vex_supports = local_puts.sort_values("weighted_vex", ascending=True).head(num_levels).reset_index(drop=True)

    _, _, wide_calls, wide_puts, wide_strike_step = get_weighted_option_data_polygon(
        ticker_symbol=ticker_symbol,
        weights=weights,
        fixed_spot=anchor_spot,
        max_distance=max_distance * 4,
    )

    combined_all = build_combined_curve(wide_calls, wide_puts)
    gamma_flip = estimate_gamma_flip(combined_all[["strike", "weighted_gex"]].copy()) if not combined_all.empty else None

    total_net_gex = float(combined_all["weighted_gex"].sum()) if not combined_all.empty else 0.0
    total_net_vex = float(combined_all["weighted_vex"].sum()) if not combined_all.empty else 0.0
    total_net_dex = float(combined_all["weighted_dex"].sum()) if not combined_all.empty else 0.0

    search_min, search_max = get_local_range(anchor_spot, max_distance, strike_step=strike_step)

    gamma_key_global = None
    if not combined_all.empty:
        combined_all["abs_weighted_gex"] = combined_all["weighted_gex"].abs()
        global_curve = combined_all.dropna(subset=["strike", "abs_weighted_gex"]).copy()
        if not global_curve.empty:
            gamma_key_global = float(global_curve.loc[global_curve["abs_weighted_gex"].idxmax(), "strike"])

    local_curve = combined_all[(combined_all["strike"] >= search_min) & (combined_all["strike"] <= search_max)].copy()

    gamma_key_local = None
    if not local_curve.empty:
        local_curve["abs_weighted_gex"] = local_curve["weighted_gex"].abs()
        local_key_df = local_curve.dropna(subset=["strike", "abs_weighted_gex"]).copy()
        if not local_key_df.empty:
            gamma_key_local = float(local_key_df.loc[local_key_df["abs_weighted_gex"].idxmax(), "strike"])

    key_level = gamma_key_local if gamma_key_local is not None else gamma_key_global
    regime = infer_gamma_regime_from_net_gex(float(live_spot), gamma_flip, total_net_gex)

    # Wall logic must match the OI chart:
    # Call Wall = strike with the largest Call OI anywhere in the visible local strike range.
    # Put Wall  = strike with the largest Put OI anywhere in the visible local strike range.
    # Do NOT use filter_local_calls/filter_local_puts here because those only keep
    # calls above spot and puts below spot. That can make the wall line disagree
    # with the OI chart when the biggest call/put concentration is on the other side of spot.
    def _wall_from_curve_column(curve_df: pd.DataFrame, oi_column: str):
        if curve_df is None or curve_df.empty or oi_column not in curve_df.columns:
            return None

        clean = curve_df[["strike", oi_column]].copy()
        clean["strike"] = pd.to_numeric(clean["strike"], errors="coerce")
        clean[oi_column] = pd.to_numeric(clean[oi_column], errors="coerce").fillna(0.0)
        clean = clean.dropna(subset=["strike"])
        clean = clean[clean[oi_column] > 0]

        if clean.empty:
            return None

        return float(clean.loc[clean[oi_column].idxmax(), "strike"])

    call_wall = _wall_from_curve_column(local_curve, "call_weighted_oi")
    put_wall = _wall_from_curve_column(local_curve, "put_weighted_oi")

    base_top_cols = [
        "strike",
        "weighted_gex",
        "total_gex",
        "weighted_vex",
        "total_vex",
        "weighted_dex",
        "total_dex",
        "total_open_interest",
        "weighted_open_interest",
        "weighted_volume",
    ]

    def safe_cols(df, cols):
        if df is None or df.empty:
            return pd.DataFrame(columns=cols)
        for c in cols:
            if c not in df.columns:
                df[c] = 0.0
        return df[cols]

    return {
        "model": "GAMMA",
        "ticker": ticker_symbol,
        "spot": round(float(live_spot), 2),
        "anchor_spot": round(float(anchor_spot), 2),
        "strike_step": float(strike_step),
        "expirations_used": expirations,
        "weights_used": weights,
        "search_range": [round(search_min, 2), round(search_max, 2)],
        "gamma_flip": gamma_flip,
        "key_level": key_level,
        "gamma_key_global": gamma_key_global,
        "gamma_key_local": gamma_key_local,
        "call_wall": call_wall,
        "put_wall": put_wall,
        "regime": regime,
        "total_net_gex": total_net_gex,
        "total_net_vex": total_net_vex,
        "total_net_dex": total_net_dex,
        "flip_source": "wide_scan" if gamma_flip is not None else "net_gex_proxy",
        "top_resistances": safe_cols(top_resistances, base_top_cols),
        "top_supports": safe_cols(top_supports, base_top_cols),
        "top_vex_resistances": safe_cols(top_vex_resistances, base_top_cols),
        "top_vex_supports": safe_cols(top_vex_supports, base_top_cols),
        "gex_curve": local_curve.to_dict(orient="records"),
        "dex_curve": local_curve[["strike", "weighted_dex", "total_dex"]].to_dict(orient="records") if not local_curve.empty else [],
        "oi_curve": local_curve[["strike", "weighted_open_interest", "total_open_interest", "call_weighted_oi", "put_weighted_oi"]].to_dict(orient="records") if not local_curve.empty else [],
        "gex_curve_wide": combined_all.to_dict(orient="records"),
    }
