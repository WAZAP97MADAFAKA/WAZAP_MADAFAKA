import pandas as pd


def nearest_level_match(level, levels, tolerance):
    if not levels:
        return None
    matches = [x for x in levels if abs(x - level) <= tolerance]
    if not matches:
        return None
    return min(matches, key=lambda x: abs(x - level))


def classify_score(score):
    if score >= 80:
        return "A+"
    if score >= 65:
        return "A"
    if score >= 50:
        return "B"
    return "SKIP"


def classify_gamma_strength(side, has_match, gamma_flip, spot, level_gex):
    if not has_match:
        return "NO_GAMMA_BACKING"

    if gamma_flip is None:
        if side == "SUPPORT":
            return "STRONG_BUT_VOLATILE" if level_gex < 0 else "GAMMA_BACKED"
        if side == "RESISTANCE":
            return "STRONG_BUT_VOLATILE" if level_gex < 0 else "GAMMA_BACKED"
        return "GAMMA_BACKED"

    # Above flip = long gamma environment
    if spot > gamma_flip:
        if side == "SUPPORT":
            if level_gex >= 0:
                return "STRONG_GAMMA_BACKED"
            return "STRONG_BUT_VOLATILE"

        if side == "RESISTANCE":
            if level_gex >= 0:
                return "STRONG_GAMMA_BACKED"
            return "WEAK_GAMMA_RESISTANCE"

    # Below flip = short gamma environment
    if spot < gamma_flip:
        if side == "SUPPORT":
            if level_gex < 0:
                return "WEAK_GAMMA_SUPPORT"
            return "WEAK_GAMMA_SUPPORT"

        if side == "RESISTANCE":
            if level_gex < 0:
                return "STRONG_GAMMA_BACKED"
            return "STRONG_BUT_VOLATILE"

    return "GAMMA_BACKED"


def get_level_gex(level, gamma_df, tolerance):
    if gamma_df is None or gamma_df.empty:
        return None

    exact = gamma_df[gamma_df["strike"] == level]
    if not exact.empty:
        row = exact.iloc[0]
        if "weighted_gex" in row:
            return float(row["weighted_gex"])

    nearby = gamma_df[(gamma_df["strike"] - level).abs() <= tolerance]
    if not nearby.empty and "weighted_gex" in nearby.columns:
        nearest = nearby.iloc[(nearby["strike"] - level).abs().argsort()].iloc[0]
        return float(nearest["weighted_gex"])

    return None


def build_confluence_from_results(ticker_symbol: str, oi: dict, gamma: dict):
    spot = gamma["spot"]
    gamma_flip = gamma["gamma_flip"]
    regime = gamma["regime"]

    tolerance = 1.0 if ticker_symbol == "SPY" else 0.5

    oi_supports = oi["top_supports"]["strike"].tolist() if not oi["top_supports"].empty else []
    oi_resistances = oi["top_resistances"]["strike"].tolist() if not oi["top_resistances"].empty else []
    gamma_supports = gamma["top_supports"]["strike"].tolist() if not gamma["top_supports"].empty else []
    gamma_resistances = gamma["top_resistances"]["strike"].tolist() if not gamma["top_resistances"].empty else []

    gamma_supports_df = gamma["top_supports"] if isinstance(gamma["top_supports"], pd.DataFrame) else pd.DataFrame(gamma["top_supports"])
    gamma_resistances_df = gamma["top_resistances"] if isinstance(gamma["top_resistances"], pd.DataFrame) else pd.DataFrame(gamma["top_resistances"])

    scored_rows = []

    for level in oi_supports:
        score = 30
        reasons = ["OI support"]

        match = nearest_level_match(level, gamma_supports, tolerance)
        has_match = match is not None
        if has_match:
            score += 30
            reasons.append(f"Gamma support nearby ({match})")

        level_gex = get_level_gex(level, gamma_supports_df, tolerance)
        if level_gex is not None:
            reasons.append(f"Level GEX {level_gex:,.0f}")

        if gamma_flip is not None and spot > gamma_flip:
            score += 15
            reasons.append("Above gamma flip")

        if abs(spot - level) <= 6:
            score += 10
            reasons.append("Near current spot")

        if oi["key_level"] is not None and abs(oi["key_level"] - level) <= tolerance:
            score += 10
            reasons.append("Near OI key level")

        gamma_strength = classify_gamma_strength(
            side="SUPPORT",
            has_match=has_match,
            gamma_flip=gamma_flip,
            spot=spot,
            level_gex=level_gex if level_gex is not None else 0.0,
        )

        if gamma_strength == "STRONG_GAMMA_BACKED":
            reasons.append("Regime and GEX support the level")
        elif gamma_strength == "STRONG_BUT_VOLATILE":
            reasons.append("Regime supports level, but local GEX is unstable")
        elif gamma_strength == "WEAK_GAMMA_SUPPORT":
            reasons.append("Gamma regime works against this support")
        elif gamma_strength == "NO_GAMMA_BACKING":
            reasons.append("No local gamma support found")

        scored_rows.append({
            "side": "SUPPORT",
            "level": level,
            "gamma_match": match,
            "level_gex": level_gex,
            "gamma_strength": gamma_strength,
            "score": score,
            "grade": classify_score(score),
            "reasons": ", ".join(reasons),
        })

    for level in oi_resistances:
        score = 30
        reasons = ["OI resistance"]

        match = nearest_level_match(level, gamma_resistances, tolerance)
        has_match = match is not None
        if has_match:
            score += 30
            reasons.append(f"Gamma resistance nearby ({match})")

        level_gex = get_level_gex(level, gamma_resistances_df, tolerance)
        if level_gex is not None:
            reasons.append(f"Level GEX {level_gex:,.0f}")

        if gamma_flip is not None and spot < gamma_flip:
            score += 15
            reasons.append("Below gamma flip")

        if abs(spot - level) <= 6:
            score += 10
            reasons.append("Near current spot")

        if oi["key_level"] is not None and abs(oi["key_level"] - level) <= tolerance:
            score += 10
            reasons.append("Near OI key level")

        gamma_strength = classify_gamma_strength(
            side="RESISTANCE",
            has_match=has_match,
            gamma_flip=gamma_flip,
            spot=spot,
            level_gex=level_gex if level_gex is not None else 0.0,
        )

        if gamma_strength == "STRONG_GAMMA_BACKED":
            reasons.append("Regime and GEX support the level")
        elif gamma_strength == "STRONG_BUT_VOLATILE":
            reasons.append("Regime supports level, but local GEX is unstable")
        elif gamma_strength == "WEAK_GAMMA_RESISTANCE":
            reasons.append("Gamma regime works against this resistance")
        elif gamma_strength == "NO_GAMMA_BACKING":
            reasons.append("No local gamma resistance found")

        scored_rows.append({
            "side": "RESISTANCE",
            "level": level,
            "gamma_match": match,
            "level_gex": level_gex,
            "gamma_strength": gamma_strength,
            "score": score,
            "grade": classify_score(score),
            "reasons": ", ".join(reasons),
        })

    scored_df = pd.DataFrame(scored_rows)
    if not scored_df.empty:
        scored_df = scored_df.sort_values(["score", "side"], ascending=[False, True]).reset_index(drop=True)

    skip_rules = [
        "Skip if price slices through the level with no rejection",
        "Skip if price reaches the level during major news",
        "Skip if the setup fights the gamma regime",
        "Skip if there is no OI + gamma confluence nearby",
        "Skip if price is too far from the next target level",
        "Skip weak support longs when below gamma flip",
        "Skip weak resistance shorts when above gamma flip",
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