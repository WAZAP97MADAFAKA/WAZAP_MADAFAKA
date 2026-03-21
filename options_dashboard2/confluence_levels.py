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


def build_confluence_from_results(ticker_symbol: str, oi: dict, gamma: dict):
    spot = gamma["spot"]
    gamma_flip = gamma["gamma_flip"]
    regime = gamma["regime"]

    tolerance = 1.0 if ticker_symbol == "SPY" else 0.5

    oi_supports = oi["top_supports"]["strike"].tolist() if not oi["top_supports"].empty else []
    oi_resistances = oi["top_resistances"]["strike"].tolist() if not oi["top_resistances"].empty else []
    gamma_supports = gamma["top_supports"]["strike"].tolist() if not gamma["top_supports"].empty else []
    gamma_resistances = gamma["top_resistances"]["strike"].tolist() if not gamma["top_resistances"].empty else []

    scored_rows = []

    for level in oi_supports:
        score = 30
        reasons = ["OI support"]

        match = nearest_level_match(level, gamma_supports, tolerance)
        if match is not None:
            score += 30
            reasons.append(f"Gamma support nearby ({match})")

        if gamma_flip is not None and spot > gamma_flip:
            score += 15
            reasons.append("Above gamma flip")

        if abs(spot - level) <= 6:
            score += 10
            reasons.append("Near current spot")

        if oi["key_level"] is not None and abs(oi["key_level"] - level) <= tolerance:
            score += 10
            reasons.append("Near OI key level")

        scored_rows.append({
            "side": "SUPPORT",
            "level": level,
            "score": score,
            "grade": classify_score(score),
            "reasons": ", ".join(reasons),
        })

    for level in oi_resistances:
        score = 30
        reasons = ["OI resistance"]

        match = nearest_level_match(level, gamma_resistances, tolerance)
        if match is not None:
            score += 30
            reasons.append(f"Gamma resistance nearby ({match})")

        if gamma_flip is not None and spot < gamma_flip:
            score += 15
            reasons.append("Below gamma flip")

        if abs(spot - level) <= 6:
            score += 10
            reasons.append("Near current spot")

        if oi["key_level"] is not None and abs(oi["key_level"] - level) <= tolerance:
            score += 10
            reasons.append("Near OI key level")

        scored_rows.append({
            "side": "RESISTANCE",
            "level": level,
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
