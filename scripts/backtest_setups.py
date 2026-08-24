"""
backtest_setups.py — Phase 0 sanity check before shipping TREND_CONTINUATION /
MOMENTUM_RALLY live in src/bot/scanner.py:compute_break_model.

Walks forward over historical 1h OHLCV, calls detect_trend_continuation() and
detect_momentum_rally() directly, applies the same confluence_score() thresholds
compute_break_model() uses live (65 for the proven setups, 70 for these two), and
resolves each fire with the same TP2 > TP1 > stop > 72h-expiry precedence app.py's
resolve_pending() uses on the live archive — so the reported win rate approximates
what would have actually been archived, not just raw detector fires.
Also runs the two already-live setups (HOT_MOMENTUM, RANGE_BOUNCE) segmented by prime
vs. off-peak session, to sanity-check the "17% off-peak WR" figure that used to gate
BUYs in scanner.py (removed 2026-08-24).

This is a same-day smell test, not a rigorous backtest (1h-bar simulation, no fees/
slippage, no partial fills) — a detector failing badly here (<35% WR) means tighten it
further before shipping; passing here is not proof of a durable edge, only that it
isn't obviously broken.

Usage:
    python scripts/backtest_setups.py
    python scripts/backtest_setups.py --days 120 --stride 6
    python scripts/backtest_setups.py --symbols BTC/USDT,ETH/USDT,SOL/USDT
"""
import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from bot.scanner import (  # noqa: E402
    DEFAULT_SYMBOLS,
    confluence_score,
    detect_hot_momentum,
    detect_momentum_rally,
    detect_range_bounce,
    detect_trend_continuation,
    fetch_crypto_ohlcv_history,
    market_regime,
)

# Same confluence thresholds compute_break_model() (scanner.py) applies live —
# without this gate, raw detector fires include weak signals the real bot would
# never archive, which understates even the already-proven setups' true WR.
CONFLUENCE_MIN = {
    "HOT_MOMENTUM": 65,
    "RANGE_BOUNCE": 65,
    "TREND_CONTINUATION": 70,
    "MOMENTUM_RALLY": 70,
}

EXPIRY_BARS = 72  # matches app.py resolve_pending(expiry_hours=72) on 1h bars
WARMUP_BARS = 210  # room for EMA200-proxy checks inside confluence_score's callers


def is_prime_hour(ts) -> bool:
    return 8 <= ts.hour < 21  # matches scanner.session_context()'s prime window


def resolve(df, fire_idx: int, entry: float, stop: float, targets: list):
    tp1 = targets[0] if len(targets) > 0 else None
    tp2 = targets[1] if len(targets) > 1 else None
    end = min(fire_idx + 1 + EXPIRY_BARS, len(df))
    for j in range(fire_idx + 1, end):
        hi = float(df["high"].iloc[j])
        lo = float(df["low"].iloc[j])
        if tp2 and hi >= tp2:
            return "WIN", (tp2 - entry) / entry * 100
        if tp1 and hi >= tp1:
            return "WIN", (tp1 - entry) / entry * 100
        if stop and lo <= stop:
            return "LOSS", (stop - entry) / entry * 100
    return "EXPIRED", None


def backtest_symbol(symbol: str, days: int, stride: int, results: dict):
    df = fetch_crypto_ohlcv_history(symbol, timeframe="1h", days=days)
    if len(df) < WARMUP_BARS + 50:
        print(f"  skip {symbol}: only {len(df)} bars (need {WARMUP_BARS + 50}+)")
        return
    print(f"  {symbol}: {len(df)} bars ({df['ts'].iloc[0].date()} → {df['ts'].iloc[-1].date()})")

    i = WARMUP_BARS
    n_fires = 0
    while i < len(df) - 1:
        window = df.iloc[: i + 1].reset_index(drop=True)
        prime = is_prime_hour(df["ts"].iloc[i])

        gain_24h = 0.0
        if i >= 24:
            prev_close = float(df["close"].iloc[i - 24])
            if prev_close:
                gain_24h = (float(df["close"].iloc[i]) - prev_close) / prev_close * 100

        candidates = [
            ("TREND_CONTINUATION", detect_trend_continuation(window, "CRYPTO")),
            ("MOMENTUM_RALLY", detect_momentum_rally(window, "CRYPTO")),
        ]
        regime = market_regime(window)
        if regime in ("RANGE", "COMPRESSION", "CHOPPY"):
            candidates.append(("RANGE_BOUNCE", detect_range_bounce(window, "CRYPTO")))
        if 15.0 <= gain_24h <= 60.0:
            candidates.append(("HOT_MOMENTUM", detect_hot_momentum(window, "CRYPTO", gain_24h)))

        fired = False
        for name, setup in candidates:
            if not setup:
                continue
            cs, _cl, _fails = confluence_score(window, None, name, setup)
            if cs < CONFLUENCE_MIN[name]:
                continue
            _direction, entry_zone, stop, targets, *_ = setup
            entry = (entry_zone[0] + entry_zone[1]) / 2
            outcome, _pct = resolve(df, i, entry, stop, targets)
            if outcome == "EXPIRED":
                continue
            bucket = "prime" if prime else "offpeak"
            results[name][bucket].append(1 if outcome == "WIN" else 0)
            fired = True
            n_fires += 1

        # Light cooldown after any fire so a multi-day persistent trend doesn't get
        # counted as dozens of near-duplicate overlapping signals (rough analogue of
        # the live bot's 36h dedup in app.py:archive_signal).
        i += 24 if fired else stride

    if n_fires:
        print(f"    → {n_fires} resolved fires")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=150, help="history window in days")
    ap.add_argument("--stride", type=int, default=6, help="bars to skip between scans when nothing fires")
    ap.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS))
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    results: dict = defaultdict(lambda: defaultdict(list))

    print(f"Backtesting {len(symbols)} symbols over {args.days} days (stride={args.stride}h)...\n")
    for sym in symbols:
        try:
            backtest_symbol(sym, args.days, args.stride, results)
        except Exception as e:
            print(f"  {sym}: FAILED — {type(e).__name__}: {e}")

    print("\n" + "=" * 72)
    print(f"{'Setup':<20}{'Session':<10}{'N':>6}{'Wins':>6}{'WR%':>8}")
    print("-" * 72)
    for name in sorted(results):
        for bucket in ("prime", "offpeak"):
            outcomes = results[name].get(bucket, [])
            if not outcomes:
                continue
            n = len(outcomes)
            wins = sum(outcomes)
            wr = wins / n * 100
            print(f"{name:<20}{bucket:<10}{n:>6}{wins:>6}{wr:>7.1f}%")
        all_outcomes = results[name]["prime"] + results[name]["offpeak"]
        if all_outcomes:
            n = len(all_outcomes)
            wins = sum(all_outcomes)
            print(f"{name:<20}{'ALL':<10}{n:>6}{wins:>6}{wins / n * 100:>7.1f}%")
    print("=" * 72)
    print("\nGate: TREND_CONTINUATION / MOMENTUM_RALLY should clear ~45% WR over ~30+ N")
    print("before being trusted; below that, tighten thresholds further before relying on them live.")


if __name__ == "__main__":
    main()
