import pandas as pd


def normalize(value, max_value):
    if max_value <= 0:
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


# 🔥 NEW: VEX strength classification
def classify_vex_strength(vex_value, max_vex):
    if vex_value is None or max_vex == 0:
        return "LOW"

    ratio = abs(vex_value) / max_vex

    if ratio >= 0.6:
        return "HIGH"
    elif ratio >= 0.3:
        return "MEDIUM"
    else:
        return "LOW"


# 🔥 NEW: Behavior engine (GEX + VEX)
def classify_behavior(gex, vex_strength):
    if gex >= 0 and vex_strength == "LOW":
        return "CLEAN"
    if gex >= 0 and vex_strength == "HIGH":
        return "FAST"
    if gex < 0 and vex_strength == "LOW":
        return "CHOP"
    if gex < 0 and vex_strength == "HIGH":
        return "EXPLOSIVE"
    return "MIXED"


# 🔥 NEW: Trade recommendation
def classify_trade(behavior):
    if behavior == "CLEAN":
        return "BOUNCE"
    if behavior == "FAST":
        return "SCALP"
    if behavior == "EXPLOSIVE":
        return "BREAKOUT"
    return "SKIP"


def build_confluence_from_results(ticker_symbol: str, oi: dict, gamma: dict):

    spot = float(gamma["spot"])

    oi_supports = pd.DataFrame(oi["top_supports"])
    oi_resistances = pd.DataFrame(oi["top_resistances"])
    gamma_supports = pd.DataFrame(gamma["top_supports"])
    gamma_resistances = pd.DataFrame(gamma["top_resistances"])

    # Max values for normalization
    max_oi = max(
        oi_supports["weighted_open_interest"].max() if not oi_supports.empty else 0,
        oi_resistances["weighted_open_interest"].max() if not oi_resistances.empty else 0,
    )

    max_gex = max(
        abs(gamma_supports["weighted_gex"]).max() if not gamma_supports.empty else 0,
        abs(gamma_resistances["weighted_gex"]).max() if not gamma_resistances.empty else 0,
    )

    max_vex = max(
        abs(gamma_supports["weighted_vex"]).max() if not gamma_supports.empty else 0,
        abs(gamma_resistances["weighted_vex"]).max() if not gamma_resistances.empty else 0,
    )

    rows = []

    for side, df in [("SUPPORT", oi_supports), ("RESISTANCE", oi_resistances)]:
        for _, row in df.iterrows():

            level = float(row["strike"])

            # Get matching gamma row
            gamma_df = gamma_supports if side == "SUPPORT" else gamma_resistances
            gamma_row = gamma_df[gamma_df["strike"] == level]

            if not gamma_row.empty:
                gex = float(gamma_row.iloc[0]["weighted_gex"])
                vex = float(gamma_row.iloc[0]["weighted_vex"])
            else:
                gex = 0.0
                vex = 0.0

            # Scores
            oi_score = normalize(row["weighted_open_interest"], max_oi) * 35
            gex_score = normalize(abs(gex), max_gex) * 30

            static_score = round(oi_score + gex_score, 1)

            # Dynamic boost (distance)
            distance = abs(spot - level)

            if ticker_symbol == "SPY":
                if distance <= 0.5:
                    proximity = 15
                elif distance <= 1:
                    proximity = 10
                elif distance <= 2:
                    proximity = 6
                else:
                    proximity = 0
            else:
                if distance <= 0.25:
                    proximity = 15
                elif distance <= 0.5:
                    proximity = 10
                elif distance <= 1:
                    proximity = 6
                else:
                    proximity = 0

            dynamic_score = min(static_score + proximity, 100)

            # 🔥 NEW logic
            vex_strength = classify_vex_strength(vex, max_vex)
            behavior = classify_behavior(gex, vex_strength)
            trade_type = classify_trade(behavior)

            rows.append({
                "side": side,
                "level": level,
                "level_gex": round(gex, 2),
                "level_vex": round(vex, 2),
                "vex_strength": vex_strength,
                "market_behavior": behavior,
                "best_trade_type": trade_type,
                "static_score": static_score,
                "dynamic_score": dynamic_score,
                "grade": classify_score(dynamic_score),
                "confidence": confidence_label(dynamic_score),
                "distance_to_spot": round(distance, 2),
            })

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values("dynamic_score", ascending=False).reset_index(drop=True)

    return {
        "levels": df,
        "spot": spot,
    }
