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


def hold_break_bias(side, gamma_strength, gamma_flip, spot):
    if gamma_strength == "STRONG_GAMMA_BACKED":
        return "LIKELY TO HOLD"
    if gamma_strength == "NO_GAMMA_BACKING":
        return "LIKELY TO BREAK"
    if gamma_strength in ["WEAK_GAMMA_SUPPORT", "WEAK_GAMMA_RESISTANCE"]:
        return "LIKELY TO BREAK"
    if gamma_strength == "STRONG_BUT_VOLATILE":
        return "CAN HOLD, BUT MESSY"

    if gamma_flip is None:
        return "NEUTRAL"

    if side == "SUPPORT":
        return "LIKELY TO HOLD" if spot > gamma_flip else "LIKELY TO BREAK"
    return "LIKELY TO HOLD" if spot < gamma_flip else "LIKELY TO BREAK"


def classify_gamma_strength(side, has_match, gamma_flip, spot, level_gex):
    if not has_match:
        return "NO_GAMMA_BACKING"

    if gamma_flip is None:
        if level_gex < 0:
            return "STRONG_BUT_VOLATILE"
        return "GAMMA_BACKED"

    if spot > gamma_flip:
        if side == "SUPPORT":
            if level_gex >= 0:
                return "STRONG_GAMMA_BACKED"
            return "STRONG_BUT_VOLATILE"
        if side == "RESISTANCE":
            if level_gex >= 0:
                return "STRONG_GAMMA_BACKED"
            return "WEAK_GAMMA_RESISTANCE"

    if spot < gamma_flip:
        if side == "SUPPORT":
            return "WEAK_GAMMA_SUPPORT"
        if side == "RESISTANCE":
            if level_gex < 0:
                return "STRONG_GAMMA_BACKED"
            return "STRONG_BUT_VOLATILE"

    return "GAMMA_BACKED"


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


def action_for_level(side, grade, hold_break_bias_value, gamma_flip, spot):
    if grade not in ["A+", "A"]:
        return "SKIP"

    if side == "SUPPORT":
        if hold_break_bias_value in ["LIKELY TO HOLD", "CAN HOLD, BUT MESSY"]:
            return "LONG"
        if gamma_flip is not None and spot < gamma_flip:
            return "SHORT"
        return "SKIP"

    if side == "RESISTANCE":
        if hold_break_bias_value in ["LIKELY TO HOLD", "CAN HOLD, BUT MESSY"]:
            return "SHORT"
        if gamma_flip is not None and spot > gamma_flip:
            return "LONG"
        return "SKIP"

    return "SKIP"


def get_proximity_score(distance_to_spot, ticker_symbol):
    """
    Dynamic part only.
    This is what lets the dynamic_score rise as price gets closer.
    """
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

    # QQQ
    if distance_to_spot <= 0.25:
        return 15.0
    if distance_to_spot <= 0.5:
        return 10.0
    if distance_to_spot <= 1.0:
        return 6.0
    if distance_to_spot <= 1.5:
        return 3.0
    return 0.0


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

    max_oi_woi = max(all_oi_woi) if all_oi_woi else 0.0
    max_gamma_abs = max(all_gamma_abs) if all_gamma_abs else 0.0

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
                # proxy regime still gives some value
                if regime == "NO_LOCAL_FLIP_LONG_GAMMA_BIAS":
                    regime_score = 18.0 if side == "SUPPORT" else 4.0
                elif regime == "NO_LOCAL_FLIP_SHORT_GAMMA_BIAS":
                    regime_score = 18.0 if side == "RESISTANCE" else 4.0
                else:
                    regime_score = 8.0

            sign_bonus = 0.0
            if level_gex is not None:
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
                    # proxy regime handling
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

            # OLD / structure-first score
            static_score = round(
                oi_strength_score + gamma_magnitude_score + regime_score + sign_bonus,
                1,
            )

            # NEW / live execution score
            dynamic_score = round(
                static_score + proximity_score + key_score,
                1,
            )

            static_score = min(static_score, 100.0)
            dynamic_score = min(dynamic_score, 100.0)

            gamma_strength = classify_gamma_strength(
                side=side,
                has_match=has_match,
                gamma_flip=gamma_flip,
                spot=spot,
                level_gex=level_gex if level_gex is not None else 0.0,
            )

            hold_break = hold_break_bias(side, gamma_strength, gamma_flip, spot)
            grade = classify_score(dynamic_score)
            static_grade = classify_score(static_score)
            action = action_for_level(side, grade, hold_break, gamma_flip, spot)

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
                    "level_gex": round(level_gex, 2) if level_gex is not None else None,
                    "level_vex": round(level_vex, 2) if level_vex is not None else None,
                    "gamma_strength": gamma_strength,
                    "static_score": static_score,
                    "static_grade": static_grade,
                    "dynamic_score": dynamic_score,
                    "grade": grade,
                    "confidence": confidence_label(dynamic_score),
                    "hold_break_bias": hold_break,
                    "action": action,
                    "distance_to_spot": round(distance_to_spot, 2),
                    "distance_to_key": round(abs(float(oi["key_level"]) - level), 2) if oi["key_level"] is not None else None,
                    "reasons": ", ".join(reasons),
                }
            )

    scored_df = pd.DataFrame(scored_rows)
    if not scored_df.empty:
        scored_df = scored_df.sort_values(["dynamic_score", "static_score", "side"], ascending=[False, False, True]).reset_index(drop=True)

    skip_rules = [
        "Skip if price slices through the level with no rejection",
        "Skip if price reaches the level during major news",
        "Skip if the setup fights the gamma regime",
        "Skip if there is no OI + gamma confluence nearby",
        "Be cautious with STRONG_BUT_VOLATILE levels because they can react sharply but not cleanly",
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
