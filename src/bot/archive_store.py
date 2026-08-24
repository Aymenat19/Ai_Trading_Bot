"""
archive_store.py — pure archive read/write/stats functions, shared by app.py
(the live scanning bot) and viewer_app.py (the read-only Streamlit Cloud dashboard).

Deliberately has no dependency on ccxt/yfinance/streamlit at module level, so the
lightweight viewer can import it without pulling in the scanner's exchange clients.
"""
import json
import time
import uuid
from datetime import datetime, timezone

ARCHIVE_PATH = "trade_archive.json"


def load_archive(path: str = ARCHIVE_PATH) -> list:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_archive(records: list, path: str = ARCHIVE_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, default=str)


def archive_signal(idea, archive: list) -> bool:
    """
    Log a new signal with three guards:
    1. 36h dedup — same symbol+action not re-logged within 36 hours (extended from 24h;
       live data showed NOM/WAXP/ONT re-entries at 31h gaps slipping through 24h window)
    2. PENDING block — never open a second position on same symbol+action while one is live
    3. Stop blacklist — 72h after stop; extended to 7 days (168h) for repeat stoppers
       (coins stopped 2+ times: SAPIEN, 1000SATS, NOM — keep them out longer)
    """
    now_ts = time.time()
    cutoff_dedup = now_ts - 36 * 3600

    # Repeat stopper: count all-time stops for this symbol to set blacklist window
    prior_stops = sum(1 for r in archive if r["symbol"] == idea.symbol and r["outcome"] == "STOPPED_OUT")
    blacklist_hours = 168 if prior_stops >= 2 else 72
    cutoff_blacklist = now_ts - blacklist_hours * 3600

    for rec in archive:
        try:
            logged_ts = datetime.fromisoformat(rec["logged_at"]).timestamp()
        except Exception:
            continue
        # Dedup: same symbol+action within 36h
        if rec["symbol"] == idea.symbol and rec["action"] == idea.action and logged_ts > cutoff_dedup:
            return False
        # PENDING block: never stack signals on same symbol+action while one is open
        if rec["symbol"] == idea.symbol and rec["action"] == idea.action and rec["outcome"] == "PENDING":
            return False
        # Blacklist: stopped out recently — window extends to 7 days for repeat stoppers
        if rec["symbol"] == idea.symbol and rec["outcome"] == "STOPPED_OUT":
            try:
                resolved_ts = datetime.fromisoformat(rec["resolved_at"]).timestamp()
                if resolved_ts > cutoff_blacklist:
                    return False
            except Exception:
                pass

    entry_low  = round(idea.entry[0], 6) if idea.entry else None
    entry_high = round(idea.entry[1], 6) if idea.entry else None
    targets    = [round(t, 6) for t in idea.targets] if idea.targets else []

    archive.append({
        "id":           str(uuid.uuid4())[:8],
        "logged_at":    datetime.now(timezone.utc).isoformat(),
        "market":       idea.market,
        "symbol":       idea.symbol,
        "action":       idea.action,
        "exchange":     ", ".join(getattr(idea, "exchanges", []) or []),
        "entry_low":    entry_low,
        "entry_high":   entry_high,
        "stop":         round(idea.stop, 6) if idea.stop else None,
        "targets":      targets,
        "tp1":          targets[0] if len(targets) > 0 else None,
        "tp2":          targets[1] if len(targets) > 1 else None,
        "rr":           round(idea.rr, 2) if idea.rr else None,
        "expected_pct": round(idea.expected_pct, 1) if idea.expected_pct else None,
        "confidence":   idea.confidence,
        "setup":        idea.why[0] if idea.why else "",
        "outcome":      "PENDING",
        "outcome_pct":  None,
        "resolved_at":  None,
        "notes":        "",
    })
    return True


def resolve_pending(archive: list, expiry_hours: int = 72) -> bool:
    """Auto-check PENDING crypto signals against latest prices. Returns True if any changed."""
    # Deferred imports: keeps this module importable (for the read-only viewer)
    # without requiring ccxt/pandas to be installed.
    import pandas as pd
    from bot.scanner import fetch_crypto_ohlcv

    now_ts  = time.time()
    updated = False

    for rec in archive:
        if rec["outcome"] != "PENDING":
            continue
        try:
            logged_ts = datetime.fromisoformat(rec["logged_at"]).timestamp()
        except Exception:
            continue

        if (now_ts - logged_ts) / 3600 > expiry_hours:
            rec["outcome"]     = "EXPIRED"
            rec["resolved_at"] = datetime.now(timezone.utc).isoformat()
            updated = True
            continue

        if rec["market"] != "CRYPTO":
            continue

        try:
            df = fetch_crypto_ohlcv(rec["symbol"], timeframe="1h", limit=48)
            if df.empty:
                continue
            # Only consider candles that opened AFTER the signal was logged —
            # TP1 is set to the 10-bar high, so earlier candles trivially exceed it.
            logged_dt = pd.Timestamp(logged_ts, unit="s", tz="UTC")
            df = df[df["ts"] > logged_dt]
            if df.empty:
                continue  # no candles after logging yet — too soon
            hi    = float(df["high"].max())
            lo    = float(df["low"].min())
            entry = rec.get("entry_low") or rec.get("entry_high") or float(df["close"].iloc[-1])
            tp1   = rec.get("tp1")
            tp2   = rec.get("tp2")
            stop  = rec.get("stop")

            def pct(price):
                return round((price - entry) / entry * 100, 1) if entry else None

            if tp2 and hi >= tp2:
                rec.update({"outcome": "HIT_TP2",      "outcome_pct": pct(tp2),  "resolved_at": datetime.now(timezone.utc).isoformat()})
            elif tp1 and hi >= tp1:
                rec.update({"outcome": "HIT_TP1",      "outcome_pct": pct(tp1),  "resolved_at": datetime.now(timezone.utc).isoformat()})
            elif stop and lo <= stop:
                rec.update({"outcome": "STOPPED_OUT",  "outcome_pct": pct(stop), "resolved_at": datetime.now(timezone.utc).isoformat()})
            else:
                continue
            updated = True
        except Exception:
            continue

    return updated


def normalize_setup_key(setup: str) -> str:
    """Group setup variants into a single display name for the reliability breakdown."""
    s = (setup or "").lower()
    if "hot momentum pullback" in s:
        return "Hot momentum pullback"
    if "range bounce" in s:
        return "Range bounce"
    if "momentum rally" in s:
        return "Momentum rally"
    if "trend continuation" in s:
        return "Trend continuation"
    if "breakout" in s:
        return "Breakout"
    if "pullback to ema" in s:
        return "Pullback to EMA"
    if "hot momentum" in s:
        return "Hot momentum (old)"
    return (setup or "Unknown")[:40]


def reliability_stats(archive: list) -> dict:
    resolved = [r for r in archive if r["outcome"] not in ("PENDING", "EXPIRED")]
    wins     = [r for r in resolved if r["outcome"] in ("HIT_TP1", "HIT_TP2", "MANUAL_WIN")]
    losses   = [r for r in resolved if r["outcome"] in ("STOPPED_OUT", "MANUAL_LOSS")]
    gains    = [r["outcome_pct"] for r in resolved if r["outcome_pct"] is not None]
    # Capture rate: how much of the expected move was actually captured at TP1
    tp1_hits = [
        r for r in resolved
        if r["outcome"] == "HIT_TP1"
        and r.get("expected_pct") and r.get("outcome_pct") is not None
        and float(r["expected_pct"]) > 0
    ]
    capture_rates = [r["outcome_pct"] / r["expected_pct"] for r in tp1_hits]
    avg_capture = round(sum(capture_rates) / len(capture_rates) * 100, 1) if capture_rates else 0.0
    return {
        "total":       len(archive),
        "resolved":    len(resolved),
        "wins":        len(wins),
        "losses":      len(losses),
        "win_rate":    round(len(wins) / len(resolved) * 100, 1) if resolved else 0.0,
        "avg_gain":    round(sum(gains) / len(gains), 1) if gains else 0.0,
        "pending":     sum(1 for r in archive if r["outcome"] == "PENDING"),
        "expired":     sum(1 for r in archive if r["outcome"] == "EXPIRED"),
        "avg_capture": avg_capture,
    }


def outcome_badge(o: str) -> str:
    return {"PENDING": "🕐 Pending", "HIT_TP1": "✅ Hit TP1", "HIT_TP2": "🎯 Hit TP2",
            "STOPPED_OUT": "❌ Stopped", "EXPIRED": "⏰ Expired",
            "MANUAL_WIN": "✅ Win", "MANUAL_LOSS": "❌ Loss"}.get(o, o)
