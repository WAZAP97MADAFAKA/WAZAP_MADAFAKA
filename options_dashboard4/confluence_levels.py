import pandas as pd


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
    if vex_value is None or max_vex is None or max_vex <= 0:
        return "LOW"

    ratio = abs(float(vex_value)) / float(max_vex)

    if ratio >= 0.60:
        return "HIGH"
    if ratio >= 0.30:
        return "MEDIUM"
    return "LOW"


def classify_gamma_strength(level_gex):
    if level_gex is None:
        return "NO_GAMMA_BACKING"

    g = float(level_gex)
    if g > 0:
        return "GAMMA_BACKED"
    if g < 0:
        return "STRONG_BUT_VOLATILE"
    return "NO_GAMMA_BACKING"


def classify_behavior(gex, vex_strength):
    if gex >= 0 and vex_strength == "LOW":
        return "CLEAN"
    if gex >= 0 and vex_strength == "MEDIUM":
        return "CONTROLLED"
    if gex >= 0 and vex_strength == "HIGH":
        return "FAST"
    if gex < 0 and vex_strength == "LOW":
        return "CHOP"
    if gex < 0 and vex_strength == "MEDIUM":
        return "UNSTABLE"
    if gex < 0 and vex_strength == "HIGH":
        return "EXPLOSIVE"
    return "MIXED"


def classify_trade_type(behavior):
    if behavior in ["CLEAN", "CONTROLLED"]:
        return "BOUNCE"
    if behavior == "FAST":
        return "SCALP"
    if behavior in ["EXPLOSIVE", "UNSTABLE"]:
        return "BREAKOUT"
    return "SKIP"


def classify_hold_break_bias(side, gex, behavior):
    if behavior in ["CLEAN", "CONTROLLED"]:
        return "LIKELY TO HOLD"
    if behavior == "FAST":
        return "CAN HOLD, BUT MESSY"
    if behavior in ["EXPLOSIVE", "UNSTABLE", "CHOP"]:
        return "LIKELY TO BREAK"
    return "NEUTRAL"


def classify_direction(side, best_trade_type, hold_break_bias):
    if best_trade_type in ["BOUNCE", "SCALP"]:
        if side == "SUPPORT":
            return "LONG"
        if side == "RESISTANCE":
            return "SHORT"

    if best_trade_type == "BREAKOUT":
        if side == "SUPPORT" and hold_break_bias == "LIKELY TO BREAK":
            return "SHORT"
        if side == "RESISTANCE" and hold_break_bias == "LIKELY TO BREAK":
            return "LONG"

    return "SKIP"


def get_proximity_score(distance, ticker_symbol):
    if ticker_symbol == "SPY":
        if distance <= 0.5:
            return 15.0
        if distance <= 1.0:
            return 10.0
        if distance <= 2.0:
            return 6.0
        if distance <= 3.0:
            return 3.0
        return 0.0

    if distance <= 0.25:
        return 15.0
    if distance <= 0.5:
        return 10.0
    if distance <= 1.0:
        return 6.0
    if distance <= 1.5:
        return 3.0
    return 0.0


def get_entry_stop_target(level, direction, best_trade_type, ticker_symbol):
    tick = 0.10 if ticker_symbol == "SPY" else 0.20
    stop_pad = 0.40 if ticker_symbol == "SPY" else 0.80

    if direction == "SKIP" or best_trade_type == "SKIP":
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


def get_probabilities(behavior, dynamic_score):
    if behavior == "CLEAN":
        bounce, breakout = 78, 22
    elif behavior == "CONTROLLED":
        bounce, breakout = 70, 30
    elif behavior == "FAST":
        bounce, breakout = 62, 38
    elif behavior == "CHOP":
        bounce, breakout = 40, 35
    elif behavior == "UNSTABLE":
        bounce, breakout = 35, 60
    elif behavior == "EXPLOSIVE":
        bounce, breakout = 20, 80
    else:
        bounce, breakout = 50, 50

    if dynamic_score >= 80:
        if bounce >= breakout:
            bounce += 3
        else:
            breakout += 3

    bounce = max(0, min(100, bounce))
    breakout = max(0, min(100, breakout))
    return bounce, breakout


def classify_trade_now_signal(dynamic_score, static_score, distance_to_spot, direction, best_trade_type, ticker_symbol):
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

    oi_supports = pd.DataFrame(oi["top_supports"])
    oi_resistances = pd.DataFrame(oi["top_resistances"])
    gamma_supports = pd.DataFrame(gamma["top_supports"])
    gamma_resistances = pd.DataFrame(gamma["top_resistances"])

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
        gamma_df = gamma_supports if side == "SUPPORT" else gamma_resistances

        for _, row in df.iterrows():
            level = float(row["strike"])

            gamma_row = gamma_df[gamma_df["strike"] == level]

            if not gamma_row.empty:
                gex = float(gamma_row.iloc[0]["weighted_gex"])
                vex = float(gamma_row.iloc[0]["weighted_vex"])
            else:
                gex = 0.0
                vex = 0.0

            oi_score = normalize(row["weighted_open_interest"], max_oi) * 35.0
            gex_score = normalize(abs(gex), max_gex) * 30.0

            # old/planning score
            static_score = round(oi_score + gex_score, 1)

            # live/proximity score
            distance = abs(spot - level)
            proximity = get_proximity_score(distance, ticker_symbol)
            dynamic_score = min(round(static_score + proximity, 1), 100.0)

            vex_strength = classify_vex_strength(vex, max_vex)
            gamma_strength = classify_gamma_strength(gex)
            market_behavior = classify_behavior(gex, vex_strength)
            best_trade_type = classify_trade_type(market_behavior)
            hold_break_bias = classify_hold_break_bias(side, gex, market_behavior)
            direction = classify_direction(side, best_trade_type, hold_break_bias)

            entry, stop, target = get_entry_stop_target(
                level=level,
                direction=direction,
                best_trade_type=best_trade_type,
                ticker_symbol=ticker_symbol,
            )

            bounce_prob, breakout_prob = get_probabilities(
                behavior=market_behavior,
                dynamic_score=dynamic_score,
            )

            trade_now_signal = classify_trade_now_signal(
                dynamic_score=dynamic_score,
                static_score=static_score,
                distance_to_spot=distance,
                direction=direction,
                best_trade_type=best_trade_type,
                ticker_symbol=ticker_symbol,
            )

            rows.append(
                {
                    "side": side,
                    "level": level,
                    "level_gex": round(gex, 2),
                    "level_vex": round(vex, 2),
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
                    "hold_break_bias": hold_break_bias,
                    "trade_now_signal": trade_now_signal,
                    "bounce_probability": bounce_prob,
                    "breakout_probability": breakout_prob,
                    "entry": entry,
                    "stop": stop,
                    "target": target,
                    "distance_to_spot": round(distance, 2),
                }
            )

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(
            ["dynamic_score", "static_score", "bounce_probability", "breakout_probability"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)

    skip_rules = [
        "Skip if price slices through the level with no rejection.",
        "Skip if price reaches the level during major news.",
        "Skip if the setup looks messy or unstable.",
        "Skip CHOP behavior unless the tape is very clear.",
    ]

    return {
        "levels": df,
        "spot": spot,
        "skip_rules": skip_rules,
    }
