import pandas as pd


def nearest_level_match(level, levels, tolerance):
    if not levels:
        return None
    matches = [x for x in levels if abs(x - level) <= tolerance]
    if not matches:
        return None
    return min(matches, key=lambda x: abs(x - level))


def safe_abs(x):
    return abs(float(x)) if x is not None else 0.0


def normalize(value, max_value):
    if max_value is None or max_value <= 0:
        return 0.0
    return min(max(float(value) / float(max_value), 0.0), 1.0)


def classify_score(score):
    if score >= 80:
        return "A+"
    if score >= 67:
        return "A"
    if score >= 52:
        return "B"
    return "SKIP"


def confidence_label(score):
    if score >= 80:
        return "HIGH"
    if score >= 67:
        return "MEDIUM"
    if score >= 52:
        return "LOW"
    return "AVOID"


def classify_vex_strength(vex_value, max_vex):
    if vex_value is None or max_vex <= 0:
        return "LOW"

    ratio = abs(vex_value) / max_vex

    if ratio >= 0.60:
        return "HIGH"
    if ratio >= 0.30:
        return "MEDIUM"
    return "LOW"


def classify_gamma_strength(side, has_match, gamma_flip, spot, level_gex, regime):
    if not has_match:
        return "NO_GAMMA_BACKING"

    if gamma_flip is None:
        if regime == "NO_LOCAL_FLIP_LONG_GAMMA_BIAS":
            if side == "SUPPORT":
                return "STRONG_GAMMA_BACKED" if level_gex >= 0 else "STRONG_BUT_VOLATILE"
            return "WEAK_GAMMA_RESISTANCE" if level_gex < 0 else "GAMMA_BACKED"

        if regime == "NO_LOCAL_FLIP_SHORT_GAMMA_BIAS":
            if side == "RESISTANCE":
                return "STRONG_GAMMA_BACKED" if level_gex < 0 else "STRONG_BUT_VOLATILE"
            return "WEAK_GAMMA_SUPPORT"

        if level_gex < 0:
            return "STRONG_BUT_VOLATILE"
        return "GAMMA_BACKED"

    if spot > gamma_flip:
        if side == "SUPPORT":
            return "STRONG_GAMMA_BACKED" if level_gex >= 0 else "STRONG_BUT_VOLATILE"
        return "STRONG_GAMMA_BACKED" if level_gex >= 0 else "WEAK_GAMMA_RESISTANCE"

    if spot < gamma_flip:
        if side == "SUPPORT":
            return "WEAK_GAMMA_SUPPORT"
        return "STRONG_GAMMA_BACKED" if level_gex < 0 else "STRONG_BUT_VOLATILE"

    return "GAMMA_BACKED"


def hold_break_bias(side, gamma_strength, gamma_flip, spot, regime):
    if gamma_strength == "STRONG_GAMMA_BACKED":
        return "LIKELY TO HOLD"
    if gamma_strength in ["NO_GAMMA_BACKING", "WEAK_GAMMA_SUPPORT", "WEAK_GAMMA_RESISTANCE"]:
        return "LIKELY TO BREAK"
    if gamma_strength == "STRONG_BUT_VOLATILE":
        return "CAN HOLD, BUT MESSY"

    if gamma_flip is None:
        if regime == "NO_LOCAL_FLIP_LONG_GAMMA_BIAS":
            return "LIKELY TO HOLD" if side == "SUPPORT" else "LIKELY TO BREAK"
        if regime == "NO_LOCAL_FLIP_SHORT_GAMMA_BIAS":
            return "LIKELY TO HOLD" if side == "RESISTANCE" else "LIKELY TO BREAK"
        return "NEUTRAL"

    if side == "SUPPORT":
        return "LIKELY TO HOLD" if spot > gamma_flip else "LIKELY TO BREAK"
    return "LIKELY TO HOLD" if spot < gamma_flip else "LIKELY TO BREAK"


def get_level_metric(level, df, metric, tolerance):
    if df is None or df.empty or metric not in df.columns:
        return None

    exact = df[df["strike"] == level]
    if not exact.empty:
        return float(exact.iloc[0][metric])

    nearby = df[(df["strike"] - level).abs() <= tolerance]
    if not nearby.empty:
        nearest = nearby.iloc[(nearby["strike"] - level).abs().argsort()].iloc[0]
        return float(nearest[metric])

    return None


def get_proximity_score(distance_to_spot, ticker_symbol):
    if ticker_symbol == "SPY":
        if distance_to_spot <= 0.5:
            return 15.0
        if distance_to_spot <= 1.0:
            return 10.0
        if distance_to_spot <= 2.0:
            return 6.0
        if distance_to_spot <= 3.0:
            return 3.0
        return 0.0

    if distance_to_spot <= 0.25:
        return 15.0
    if distance_to_spot <= 0.5:
        return 10.0
    if distance_to_spot <= 1.0:
        return 6.0
    if distance_to_spot <= 1.5:
        return 3.0
    return 0.0


def classify_market_behavior(level_gex, vex_strength):
    if level_gex >= 0 and vex_strength == "LOW":
        return "CLEAN"
    if level_gex >= 0 and vex_strength == "MEDIUM":
        return "CONTROLLED"
    if level_gex >= 0 and vex_strength == "HIGH":
        return "FAST"
    if level_gex < 0 and vex_strength == "LOW":
        return "CHOP"
    if level_gex < 0 and vex_strength == "MEDIUM":
        return "UNSTABLE"
    if level_gex < 0 and vex_strength == "HIGH":
        return "EXPLOSIVE"
    return "MIXED"


def classify_best_trade_type(market_behavior, hold_break_bias_value):
    if market_behavior == "CLEAN":
        return "BOUNCE"
    if market_behavior == "CONTROLLED":
        return "BOUNCE"
    if market_behavior == "FAST":
        return "SCALP"
    if market_behavior == "EXPLOSIVE":
        return "BREAKOUT"
    if market_behavior == "UNSTABLE":
        return "BREAKOUT" if hold_break_bias_value == "LIKELY TO BREAK" else "SKIP"
    return "SKIP"


def classify_direction(side, best_trade_type, hold_break_bias_value):
    if best_trade_type in ["BOUNCE", "SCALP"]:
        if side == "SUPPORT":
            return "LONG"
        if side == "RESISTANCE":
            return "SHORT"
    if best_trade_type == "BREAKOUT":
        if side == "SUPPORT" and hold_break_bias_value == "LIKELY TO BREAK":
            return "SHORT"
        if side == "RESISTANCE" and hold_break_bias_value == "LIKELY TO BREAK":
            return "LONG"
    return "SKIP"


def get_entry_stop_target(level, side, best_trade_type, direction, ticker_symbol):
    tick = 0.10 if ticker_symbol == "SPY" else 0.20
    stop_pad = 0.40 if ticker_symbol == "SPY" else 0.80

    if direction == "SKIP":
        return None, None, None

    if best_trade_type in ["BOUNCE", "SCALP"]:
        if direction == "LONG":
            entry = level + tick
            stop = level - stop_pad
            risk = entry - stop
            target = entry + (1.2 * risk if best_trade_type == "SCALP" else 2.0 * risk)
        else:
            entry = level - tick
            stop = level + stop_pad
            risk = stop - entry
            target = entry - (1.2 * risk if best_trade_type == "SCALP" else 2.0 * risk)
    else:
        # BREAKOUT
        if direction == "LONG":
            entry = level + tick
            stop = level - (stop_pad * 0.75)
            risk = entry - stop
            target = entry + 2.5 * risk
        else:
            entry = level - tick
            stop = level + (stop_pad * 0.75)
            risk = stop - entry
            target = entry - 2.5 * risk

    return round(entry, 2), round(stop, 2), round(target, 2)


def get_probabilities(market_behavior, hold_break_bias_value, gamma_strength, dynamic_score):
    bounce = 50
    breakout = 50

    if market_behavior == "CLEAN":
        bounce, breakout = 78, 22
    elif market_behavior == "CONTROLLED":
        bounce, breakout = 70, 30
    elif market_behavior == "FAST":
        bounce, breakout = 62, 38
    elif market_behavior == "CHOP":
        bounce, breakout = 40, 35
    elif market_behavior == "UNSTABLE":
        bounce, breakout = 35, 60
    elif market_behavior == "EXPLOSIVE":
        bounce, breakout = 20, 80

    if hold_break_bias_value == "LIKELY TO HOLD":
        bounce += 8
        breakout -= 8
    elif hold_break_bias_value == "LIKELY TO BREAK":
        breakout += 8
        bounce -= 8

    if gamma_strength == "STRONG_GAMMA_BACKED":
        bounce += 5
    elif gamma_strength in ["WEAK_GAMMA_SUPPORT", "WEAK_GAMMA_RESISTANCE", "NO_GAMMA_BACKING"]:
        breakout += 5

    if dynamic_score >= 80:
        bounce += 3 if bounce >= breakout else 0
        breakout += 3 if breakout > bounce else 0

    bounce = max(0, min(100, bounce))
    breakout = max(0, min(100, breakout))
    return bounce, breakout


def classify_trade_now_signal(dynamic_score, static_score, distance_to_spot, best_trade_type, direction, ticker_symbol):
    close_threshold = 1.0 if ticker_symbol == "SPY" else 0.5
    delta = dynamic_score - static_score

    if direction == "SKIP" or best_trade_type == "SKIP":
        return "SKIP"

    if dynamic_score >= 80 and delta >= 6 and distance_to_spot <= close_threshold:
        return "TRADE NOW"

    if dynamic_score >= 67 and distance_to_spot <= close_threshold * 2:
        return "WATCH"

    return "PLAN"


def build_confluence_from_results(ticker_symbol: str, oi: dict, gamma: dict):
    spot = float(gamma["spot"])
    gamma_flip = gamma["gamma_flip"]
    regime = gamma["regime"]

    tolerance = 1.0 if ticker_symbol == "SPY" else 0.5

    oi_supports_df = oi["top_supports"] if isinstance(oi["top_supports"], pd.DataFrame) else pd.DataFrame(oi["top_supports"])
    oi_resistances_df = oi["top_resistances"] if isinstance(oi["top_resistances"], pd.DataFrame) else pd.DataFrame(oi["top_resistances"])
    gamma_supports_df = gamma["top_supports"] if isinstance(gamma["top_supports"], pd.DataFrame) else pd.DataFrame(gamma["top_supports"])
    gamma_resistances_df = gamma["top_resistances"] if isinstance(gamma["top_resistances"], pd.DataFrame) else pd.DataFrame(gamma["top_resistances"])

    oi_supports = oi_supports_df["strike"].tolist() if not oi_supports_df.empty else []
    oi_resistances = oi_resistances_df["strike"].tolist() if not oi_resistances_df.empty else []
    gamma_supports = gamma_supports_df["strike"].tolist() if not gamma_supports_df.empty else []
    gamma_resistances = gamma_resistances_df["strike"].tolist() if not gamma_resistances_df.empty else []

    all_oi_woi = []
    if not oi_supports_df.empty and "weighted_open_interest" in oi_supports_df.columns:
        all_oi_woi.extend(oi_supports_df["weighted_open_interest"].astype(float).tolist())
    if not oi_resistances_df.empty and "weighted_open_interest" in oi_resistances_df.columns:
        all_oi_woi.extend(oi_resistances_df["weighted_open_interest"].astype(float).tolist())

    all_gamma_abs = []
    if not gamma_supports_df.empty and "weighted_gex" in gamma_supports_df.columns:
        all_gamma_abs.extend(gamma_supports_df["weighted_gex"].abs().astype(float).tolist())
    if not gamma_resistances_df.empty and "weighted_gex" in gamma_resistances_df.columns:
        all_gamma_abs.extend(gamma_resistances_df["weighted_gex"].abs().astype(float).tolist())

    all_vex_abs = []
    if not gamma_supports_df.empty and "weighted_vex" in gamma_supports_df.columns:
        all_vex_abs.extend(gamma_supports_df["weighted_vex"].abs().astype(float).tolist())
    if not gamma_resistances_df.empty and "weighted_vex" in gamma_resistances_df.columns:
        all_vex_abs.extend(gamma_resistances_df["weighted_vex"].abs().astype(float).tolist())

    max_oi_woi = max(all_oi_woi) if all_oi_woi else 0.0
    max_gamma_abs = max(all_gamma_abs) if all_gamma_abs else 0.0
    max_vex_abs = max(all_vex_abs) if all_vex_abs else 0.0

    scored_rows = []

    for side, oi_df, level_list, gamma_list, gamma_df in [
        ("SUPPORT", oi_supports_df, oi_supports, gamma_supports, gamma_supports_df),
        ("RESISTANCE", oi_resistances_df, oi_resistances, gamma_resistances, gamma_resistances_df),
    ]:
        for level in level_list:
            row = oi_df[oi_df["strike"] == level].iloc[0]

            oi_weighted = float(row.get("weighted_open_interest", 0.0))
            oi_strength_score = normalize(oi_weighted, max_oi_woi) * 35.0

            match = nearest_level_match(level, gamma_list, tolerance)
            has_match = match is not None

            level_gex = get_level_metric(level, gamma_df, "weighted_gex", tolerance)
            level_vex = get_level_metric(level, gamma_df, "weighted_vex", tolerance)

            level_gex = 0.0 if level_gex is None else float(level_gex)
            level_vex = 0.0 if level_vex is None else float(level_vex)

            gamma_magnitude_score = normalize(safe_abs(level_gex), max_gamma_abs) * 30.0 if has_match else 0.0

            regime_score = 0.0
            if gamma_flip is not None:
                if side == "SUPPORT" and spot > gamma_flip:
                    regime_score = 18.0
                elif side == "RESISTANCE" and spot < gamma_flip:
                    regime_score = 18.0
                else:
                    regime_score = 4.0
            else:
                if regime == "NO_LOCAL_FLIP_LONG_GAMMA_BIAS":
                    regime_score = 18.0 if side == "SUPPORT" else 4.0
                elif regime == "NO_LOCAL_FLIP_SHORT_GAMMA_BIAS":
                    regime_score = 18.0 if side == "RESISTANCE" else 4.0
                else:
                    regime_score = 8.0

            sign_bonus = 0.0
            if gamma_flip is not None:
                if side == "SUPPORT":
                    if spot > gamma_flip and level_gex >= 0:
                        sign_bonus = 10.0
                    elif spot > gamma_flip and level_gex < 0:
                        sign_bonus = 5.0
                    elif spot < gamma_flip and level_gex < 0:
                        sign_bonus = -10.0
                elif side == "RESISTANCE":
                    if spot < gamma_flip and level_gex < 0:
                        sign_bonus = 10.0
                    elif spot < gamma_flip and level_gex >= 0:
                        sign_bonus = 5.0
                    elif spot > gamma_flip and level_gex < 0:
                        sign_bonus = -10.0
            else:
                if regime == "NO_LOCAL_FLIP_LONG_GAMMA_BIAS":
                    if side == "SUPPORT" and level_gex >= 0:
                        sign_bonus = 10.0
                    elif side == "SUPPORT" and level_gex < 0:
                        sign_bonus = 5.0
                    elif side == "RESISTANCE" and level_gex < 0:
                        sign_bonus = -10.0
                elif regime == "NO_LOCAL_FLIP_SHORT_GAMMA_BIAS":
                    if side == "RESISTANCE" and level_gex < 0:
                        sign_bonus = 10.0
                    elif side == "RESISTANCE" and level_gex >= 0:
                        sign_bonus = 5.0
                    elif side == "SUPPORT" and level_gex < 0:
                        sign_bonus = -10.0

            distance_to_spot = abs(spot - level)
            proximity_score = get_proximity_score(distance_to_spot, ticker_symbol)

            key_score = 0.0
            if oi["key_level"] is not None:
                dist_to_key = abs(float(oi["key_level"]) - level)
                if dist_to_key <= tolerance:
                    key_score = 7.0
                elif dist_to_key <= tolerance * 2:
                    key_score = 3.0

            static_score = round(
                oi_strength_score + gamma_magnitude_score + regime_score + sign_bonus,
                1,
            )
            dynamic_score = round(static_score + proximity_score + key_score, 1)

            static_score = min(static_score, 100.0)
            dynamic_score = min(dynamic_score, 100.0)

            vex_strength = classify_vex_strength(level_vex, max_vex_abs)
            gamma_strength = classify_gamma_strength(
                side=side,
                has_match=has_match,
                gamma_flip=gamma_flip,
                spot=spot,
                level_gex=level_gex,
                regime=regime,
            )
            hold_break = hold_break_bias(side, gamma_strength, gamma_flip, spot, regime)
            market_behavior = classify_market_behavior(level_gex, vex_strength)
            best_trade_type = classify_best_trade_type(market_behavior, hold_break)
            direction = classify_direction(side, best_trade_type, hold_break)
            entry, stop, target = get_entry_stop_target(level, side, best_trade_type, direction, ticker_symbol)
            bounce_prob, breakout_prob = get_probabilities(
                market_behavior,
                hold_break,
                gamma_strength,
                dynamic_score,
            )
            trade_now_signal = classify_trade_now_signal(
                dynamic_score=dynamic_score,
                static_score=static_score,
                distance_to_spot=distance_to_spot,
                best_trade_type=best_trade_type,
                direction=direction,
                ticker_symbol=ticker_symbol,
            )

            reasons = [f"OI {oi_weighted:,.0f}"]
            if has_match:
                reasons.append(f"Gamma match {match}")
            if level_gex is not None:
                reasons.append(f"GEX {level_gex:,.0f}")
            if level_vex is not None:
                reasons.append(f"VEX {level_vex:,.0f}")
            if proximity_score > 0:
                reasons.append(f"Near price +{proximity_score}")
            if key_score > 0:
                reasons.append(f"Near key +{key_score}")

            scored_rows.append(
                {
                    "side": side,
                    "level": level,
                    "gamma_match": match,
                    "level_gex": round(level_gex, 2),
                    "level_vex": round(level_vex, 2),
                    "vex_strength": vex_strength,
                    "gamma_strength": gamma_strength,
                    "market_behavior": market_behavior,
                    "best_trade_type": best_trade_type,
                    "direction": direction,
                    "static_score": static_score,
                    "static_grade": classify_score(static_score),
                    "dynamic_score": dynamic_score,
                    "grade": classify_score(dynamic_score),
                    "confidence": confidence_label(dynamic_score),
                    "hold_break_bias": hold_break,
                    "trade_now_signal": trade_now_signal,
                    "bounce_probability": bounce_prob,
                    "breakout_probability": breakout_prob,
                    "entry": entry,
                    "stop": stop,
                    "target": target,
                    "distance_to_spot": round(distance_to_spot, 2),
                    "distance_to_key": round(abs(float(oi["key_level"]) - level), 2) if oi["key_level"] is not None else None,
                    "reasons": ", ".join(reasons),
                }
            )

    scored_df = pd.DataFrame(scored_rows)
    if not scored_df.empty:
        scored_df = scored_df.sort_values(
            ["dynamic_score", "static_score", "bounce_probability", "breakout_probability"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)

    skip_rules = [
        "Skip if price slices through the level with no rejection.",
        "Skip if price reaches the level during major news.",
        "Skip if the setup fights the gamma regime.",
        "Skip if there is no OI + gamma confluence nearby.",
        "Be cautious with STRONG_BUT_VOLATILE levels because they can react sharply but not cleanly.",
        "Skip CHOP behavior unless a very clear tape confirmation appears.",
    ]

    return {
        "ticker": ticker_symbol,
        "spot": spot,
        "oi_fixed_spot": oi.get("spot"),
        "gamma_flip": gamma_flip,
        "regime": regime,
        "oi_key_level": oi["key_level"],
        "gamma_key_level": gamma["key_level"],
        "levels": scored_df,
        "skip_rules": skip_rules,
    }
