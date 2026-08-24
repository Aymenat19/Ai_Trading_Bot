"""
diagnose.py — run this from your project root to find out exactly why signals aren't showing.

Usage:
    python diagnose.py

It will test:
1. ccxt / yfinance imports
2. Binance data fetch for a few symbols
3. Indicator calculation
4. Setup detection
5. Final signal output
"""
import sys, os, traceback

sys.path.insert(0, os.path.abspath("src"))

TEST_SYMBOLS = ["BTC/USDT", "ETH/USDT", "DOGE/USDT"]
TIMEFRAME = "1h"
LIMIT = 500

SEP = "─" * 60

def ok(msg):  print(f"  ✅  {msg}")
def fail(msg): print(f"  ❌  {msg}")
def warn(msg): print(f"  ⚠️   {msg}")
def section(title): print(f"\n{SEP}\n  {title}\n{SEP}")

# ── 1. Imports ────────────────────────────────────────────────────────────────
section("1. Checking imports")
try:
    import ccxt
    ok(f"ccxt {ccxt.__version__}")
except Exception as e:
    fail(f"ccxt import failed: {e}")
    sys.exit(1)

try:
    import yfinance as yf
    ok("yfinance OK")
except Exception as e:
    warn(f"yfinance not available (stocks won't work): {e}")

try:
    import pandas as pd
    import numpy as np
    ok(f"pandas {pd.__version__}, numpy {np.__version__}")
except Exception as e:
    fail(f"pandas/numpy: {e}")
    sys.exit(1)

try:
    from bot.scanner import (
        fetch_crypto_ohlcv, compute_break_model, market_regime,
        rsi, atr, adx, ema, scan_spot_and_stocks
    )
    ok("bot.scanner imported successfully")
except Exception as e:
    fail(f"bot.scanner import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── 2. Data fetch ─────────────────────────────────────────────────────────────
section("2. Testing Binance data fetch")
dfs = {}
for sym in TEST_SYMBOLS:
    try:
        df = fetch_crypto_ohlcv(sym, timeframe=TIMEFRAME, limit=LIMIT)
        if df.empty:
            fail(f"{sym}: returned EMPTY dataframe")
        elif len(df) < 50:
            warn(f"{sym}: only {len(df)} bars returned (need 50+)")
            dfs[sym] = df
        else:
            ok(f"{sym}: {len(df)} bars, last close = {df['close'].iloc[-1]:.6f}")
            dfs[sym] = df
    except Exception as e:
        fail(f"{sym}: fetch failed — {type(e).__name__}: {e}")
        traceback.print_exc()

if not dfs:
    fail("No symbols fetched successfully. Check your internet connection and ccxt version.")
    sys.exit(1)

# ── 3. Indicators ─────────────────────────────────────────────────────────────
section("3. Testing indicators on BTC/USDT")
sym = "BTC/USDT" if "BTC/USDT" in dfs else list(dfs.keys())[0]
df = dfs[sym]

try:
    rsi_val = float(rsi(df["close"], 14).iloc[-1])
    ok(f"RSI(14) = {rsi_val:.2f}  (expect 0–100, suspicious if NaN)")
except Exception as e:
    fail(f"RSI failed: {e}")

try:
    atr_val = float(atr(df, 14).iloc[-1])
    ok(f"ATR(14) = {atr_val:.6f}")
except Exception as e:
    fail(f"ATR failed: {e}")

try:
    adx_val = float(adx(df, 14).iloc[-1])
    ok(f"ADX(14) = {adx_val:.2f}  (>20 = trending, <20 = weak)")
except Exception as e:
    fail(f"ADX failed: {e}")

try:
    e20_val = float(ema(df["close"], 20).iloc[-1])
    e50_val = float(ema(df["close"], 50).iloc[-1])
    ok(f"EMA20 = {e20_val:.4f},  EMA50 = {e50_val:.4f}")
    if e20_val > e50_val:
        ok("EMA20 > EMA50 → short-term uptrend")
    else:
        warn("EMA20 < EMA50 → short-term downtrend (fewer BUY signals expected)")
except Exception as e:
    fail(f"EMA failed: {e}")

# ── 4. Regime ─────────────────────────────────────────────────────────────────
section("4. Market regime detection")
for sym, df in dfs.items():
    try:
        regime = market_regime(df)
        ok(f"{sym}: regime = {regime}")
        if regime in ("CHOPPY", "COMPRESSION"):
            warn(f"  → {sym} is {regime}: fewer trend setups will fire. This is normal.")
    except Exception as e:
        fail(f"{sym} regime: {e}")

# ── 5. Full signal ────────────────────────────────────────────────────────────
section("5. Full signal detection (compute_break_model)")
for sym, df in dfs.items():
    try:
        result = compute_break_model(df, market="CRYPTO")
        direction, entry, stop, targets, rr, exp_pct, conf, why, liq, vol = result
        print(f"\n  {sym}:")
        print(f"    Direction : {direction}")
        print(f"    Entry     : {entry}")
        print(f"    Stop      : {stop}")
        print(f"    Targets   : {targets}")
        print(f"    R:R       : {rr}")
        print(f"    Exp %     : {exp_pct}")
        print(f"    Confidence: {conf}%")
        print(f"    Why       :")
        for w in why:
            print(f"      • {w}")
        if direction == "NONE":
            warn(f"  → No setup found for {sym}. See 'Why' above for what's missing.")
        else:
            ok(f"  → Setup found: {direction} with {conf}% confidence")
    except Exception as e:
        fail(f"{sym} compute_break_model failed: {e}")
        traceback.print_exc()

# ── 6. Holdings check ─────────────────────────────────────────────────────────
section("6. Holdings file check")
try:
    import json
    with open("holdings.json", "r") as f:
        holdings = json.load(f)
    crypto_held = list(holdings.get("crypto", {}).keys())
    stocks_held = list(holdings.get("stocks", {}).keys())
    ok(f"holdings.json loaded: {len(crypto_held)} crypto, {len(stocks_held)} stocks")
    if crypto_held:
        print(f"    Crypto holdings: {crypto_held}")
    if stocks_held:
        print(f"    Stock holdings : {stocks_held}")
except FileNotFoundError:
    warn("holdings.json not found — will use empty holdings (all signals show as new entries)")
except Exception as e:
    fail(f"holdings.json error: {e}")

# ── 7. End-to-end scan ────────────────────────────────────────────────────────
section("7. End-to-end scan (3 symbols)")
try:
    ideas = scan_spot_and_stocks(
        crypto_symbols=TEST_SYMBOLS,
        stock_tickers=[],
        timeframe=TIMEFRAME,
        limit=LIMIT,
    )
    for idea in ideas:
        action_icon = {"BUY": "🟢", "SELL": "🔴", "WATCH": "🟡", "HOLD": "🔵", "AVOID": "⚫"}.get(idea.action, "❓")
        print(f"  {action_icon}  {idea.symbol:15s}  {idea.action:6s}  conf={idea.confidence}%  rr={idea.rr}  exp={idea.expected_pct}%")
        if idea.why:
            print(f"       → {idea.why[0]}")
except Exception as e:
    fail(f"scan_spot_and_stocks failed: {e}")
    traceback.print_exc()

print(f"\n{SEP}")
print("  Diagnosis complete. Share the output above if you need further help.")
print(SEP)
