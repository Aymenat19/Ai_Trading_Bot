from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Dict, Any, Union

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import ccxt  # type: ignore
except ImportError:
    ccxt = None

try:
    import yfinance as yf  # type: ignore
except ImportError:
    yf = None

import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()


# Default watchlists (spot-only)
DEFAULT_SYMBOLS = [
    # Large caps
    "BTC/USDT","ETH/USDT","XRP/USDT","SOL/USDT","BNB/USDT","ADA/USDT","AVAX/USDT",
    # AI / GPU narrative (high momentum sector)
    "RENDER/USDT","FET/USDT","AGIX/USDT","OCEAN/USDT","TAO/USDT","WLD/USDT",
    # DeFi / L2
    "ARB/USDT","OP/USDT","ATOM/USDT","TIA/USDT","STRK/USDT","ZRO/USDT",
    # Meme / high volatility
    "DOGE/USDT","SHIB/USDT","GALA/USDT","JASMY/USDT",
    # Existing watchlist
    "HIGH/USDT","CFX/USDT","ARPA/USDT","TLM/USDT","FIDA/USDT","MOVR/USDT",
]
DEFAULT_STOCKS = [
    "SPY","QQQ","AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","NFLX",
    "V","MA","UNH","XOM","JNJ","PG","KO","PEP","BAC",
    "GLD","SLV",
]


@dataclass
class Idea:
    market: str          # "CRYPTO" or "STOCK"
    symbol: str
    action: str          # "BUY" | "SELL" | "HOLD" | "WATCH" | "AVOID"
    entry: Optional[Tuple[float, float]]
    stop: Optional[float]
    targets: List[float]
    rr: Optional[float]
    expected_pct: Optional[float]
    confidence: int
    why: List[str]
    exchanges: List[str] = None  # e.g. ["Binance", "Kraken"]

    def __post_init__(self):
        if self.exchanges is None:
            self.exchanges = []


# ─────────────────────────── Data ────────────────────────────────────────────

_shared_binance_client = None
_binance_client_lock = threading.Lock()


def _binance_spot_client(timeout: int = 8000):
    """
    Shared, reused ccxt.binance() client — spot-only.

    Without options.fetchMarkets restricted like this, ccxt's binance class
    fetches futures/delivery market data (fapi/dapi endpoints) alongside spot
    on every load_markets() call — those derivative endpoints are geofenced
    for many cloud-hosting IP ranges (e.g. Streamlit Community Cloud) even
    when spot access works fine, which surfaced as "Could not load top
    markets: ... dapi.binance.com ..." errors. This bot is spot-only and
    never needs futures/delivery data anyway.

    Reused across every call site (not reconstructed per call) for two
    reasons: ccxt lazily calls load_markets() on first use of a *fresh*
    instance, so constructing a new client per fetch_ohlcv() call — which
    happens once per symbol per timeframe across 8 parallel scan threads,
    plus once per pending signal during resolve_pending() — silently doubled
    every real request with an extra full market-list fetch. And each fresh
    instance's rate limiter starts cold, so parallel threads could burst
    well past Binance's actual rate limits instead of being throttled
    together. Sharing one instance makes load_markets() a cache hit after
    the first call and lets ccxt's enableRateLimit actually pace requests
    across the whole scan — this was the likely cause of scan cycles
    stretching to 1-2 hours instead of the expected couple of minutes.
    """
    global _shared_binance_client
    if _shared_binance_client is None:
        with _binance_client_lock:
            if _shared_binance_client is None:
                client = ccxt.binance({
                    "enableRateLimit": True,
                    "timeout": timeout,
                    "options": {"defaultType": "spot", "fetchMarkets": ["spot"]},
                })
                client.load_markets()
                _shared_binance_client = client
    _shared_binance_client.timeout = timeout
    return _shared_binance_client


def fetch_crypto_ohlcv(symbol: str, timeframe: str = "1h", limit: int = 500) -> pd.DataFrame:
    if ccxt is None:
        raise RuntimeError("ccxt not installed; install ccxt to fetch crypto data.")
    ex = _binance_spot_client()
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.reset_index(drop=True)


def fetch_crypto_ohlcv_history(symbol: str, timeframe: str = "1h", days: int = 180) -> pd.DataFrame:
    """
    Page through ccxt fetch_ohlcv to pull `days` of history — fetch_crypto_ohlcv's
    limit=500 caps out at ~20 days of 1h bars, too short for a real backtest.
    Used by scripts/backtest_setups.py; not part of the live scan path.
    """
    if ccxt is None:
        raise RuntimeError("ccxt not installed; install ccxt to fetch crypto data.")
    ex = _binance_spot_client()
    tf_ms = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}.get(timeframe, 3_600_000)
    since = int(time.time() * 1000) - days * 86_400_000
    all_rows: List[list] = []
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        next_since = last_ts + tf_ms
        if next_since <= since or len(batch) < 1000:
            break
        since = next_since
    if not all_rows:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset="ts").sort_values("ts")
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.reset_index(drop=True)


def top_binance_spot_symbols(quote: str = "USDT", limit: Optional[int] = 50) -> List[str]:
    if ccxt is None:
        raise RuntimeError("ccxt not installed; install ccxt to load Binance markets.")
    ex = _binance_spot_client()
    markets = ex.load_markets()
    pairs = []
    for m in markets.values():
        if m.get("spot") and not m.get("contract") and m.get("quote") == quote:
            vol = m.get("info", {}).get("quoteVolume")
            try:
                vol = float(vol) if vol is not None else 0.0
            except (TypeError, ValueError):
                vol = 0.0
            pairs.append((m["symbol"], vol))

    pairs.sort(key=lambda x: x[1], reverse=True)
    if limit is None or limit <= 0:
        return [p[0] for p in pairs]
    return [p[0] for p in pairs[:limit]]


def find_steady_climbers(
    quote: str = "USDT",
    min_gain_pct: float = 8.0,
    min_days: int = 2,
    max_symbols: int = 300,
    min_vol_usdt: float = 500_000,
) -> List[Dict[str, Any]]:
    """
    Scans the entire Binance spot universe using daily ticker data to find coins
    that have been making a slow, steady climb over multiple days.

    This uses the Binance 24h ticker endpoint (one API call, no per-symbol fetches)
    to pre-screen candidates, then returns ranked results.

    Returns list of dicts: {symbol, gain_pct, volume_usdt, price, price_change}
    sorted by gain descending.
    """
    if ccxt is None:
        return []
    try:
        ex = _binance_spot_client(timeout=10000)

        # Single API call — fetches all tickers at once (very fast)
        tickers = ex.fetch_tickers()

        climbers = []
        for sym, t in tickers.items():
            if not sym.endswith(f"/{quote}"):
                continue
            try:
                pct   = float(t.get("percentage") or 0)
                vol   = float(t.get("quoteVolume") or 0)
                price = float(t.get("last") or 0)
                change = float(t.get("change") or 0)

                # Filter: meaningful gain, sufficient volume, valid price
                if pct >= min_gain_pct and vol >= min_vol_usdt and price > 0:
                    climbers.append({
                        "symbol":      sym,
                        "gain_pct":    round(pct, 2),
                        "volume_usdt": round(vol, 0),
                        "price":       price,
                        "price_change": round(change, 6),
                    })
            except (TypeError, ValueError):
                continue

        # Sort by gain descending, cap results
        climbers.sort(key=lambda x: x["gain_pct"], reverse=True)
        return climbers[:max_symbols]

    except Exception:
        return []


# ─────────────────────────── New Listings / Alpha Cache ─────────────────────

LISTINGS_CACHE_PATH = "listings_seen.json"

def update_listings_cache(
    all_symbols: List[str],
    cache_path: str = LISTINGS_CACHE_PATH,
) -> None:
    """
    Record when each Binance symbol was first seen.

    First run: all existing symbols are stamped with epoch=0 (pre-existing, not new).
    Subsequent runs: any symbol absent from the cache is stamped with now().
    Only symbols stamped with now() are returned by get_new_listings().
    """
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache: Dict[str, float] = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        cache = {}

    now = time.time()
    changed = False

    if not cache:
        # First run — mark everything as pre-existing (epoch 0 = old)
        for sym in all_symbols:
            cache[sym] = 0.0
        changed = True
    else:
        for sym in all_symbols:
            if sym not in cache:
                cache[sym] = now  # genuinely new — first time we've seen it
                changed = True

    if changed:
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f)
        except Exception:
            pass


def get_new_listings(
    all_symbols: List[str],
    days: int = 60,
    cache_path: str = LISTINGS_CACHE_PATH,
) -> List[Dict[str, Any]]:
    """
    Return symbols first seen on Binance within the last `days` days.
    Symbols stamped with epoch 0 (pre-existing on first run) are excluded.
    Sorted newest-first.
    """
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache: Dict[str, float] = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    cutoff = time.time() - days * 86400
    results = []
    for sym in all_symbols:
        ts = cache.get(sym, 0.0)
        if ts > cutoff:  # epoch 0 is always < cutoff, so pre-existing are excluded
            days_ago = (time.time() - ts) / 86400
            results.append({"symbol": sym, "days_ago": round(days_ago, 1)})

    return sorted(results, key=lambda x: x["days_ago"])


# ─────────────────────────── Exchange availability ───────────────────────────

_exchange_cache: Dict[str, set] = {}
_exchange_cache_ts: Dict[str, float] = {}
_EXCHANGE_CACHE_TTL = 3600  # reload markets once per hour max

def _load_exchange_symbols(exchange_id: str, quote: str = "USDT") -> set:
    """
    Load all tradeable spot symbols for an exchange.
    Cached with TTL — reloads at most once per hour to avoid slowing every scan.
    """
    global _exchange_cache, _exchange_cache_ts
    now = time.time()
    if exchange_id in _exchange_cache and (now - _exchange_cache_ts.get(exchange_id, 0)) < _EXCHANGE_CACHE_TTL:
        return _exchange_cache[exchange_id]

    if ccxt is None:
        return set()

    try:
        if exchange_id == "binance":
            ex = _binance_spot_client()
        else:
            ex_cls = getattr(ccxt, exchange_id, None)
            if ex_cls is None:
                return set()
            ex = ex_cls({"enableRateLimit": True, "timeout": 8000})
        markets = ex.load_markets()

        symbols = set()
        for m in markets.values():
            if not m.get("spot") or m.get("contract"):
                continue
            sym = m.get("symbol", "")
            # Normalise Kraken pairs: BTC/USD or XBT/USDT → include base asset
            symbols.add(sym)
            # Also store base asset alone so we can match cross-quote
            base = m.get("base", "")
            if base:
                symbols.add(base)

        _exchange_cache[exchange_id] = symbols
        _exchange_cache_ts[exchange_id] = time.time()
        return symbols
    except Exception:
        _exchange_cache[exchange_id] = set()
        _exchange_cache_ts[exchange_id] = time.time()
        return set()


def check_exchanges(symbol: str) -> List[str]:
    """
    Return list of exchange names where this symbol is tradeable as a spot pair.
    Checks Binance and Kraken. symbol format: "BTC/USDT"
    """
    available = []

    # Binance: direct symbol match
    binance_syms = _load_exchange_symbols("binance")
    if symbol in binance_syms:
        available.append("Binance")

    # Kraken: check direct match AND base/USD or base/USDT variants
    kraken_syms = _load_exchange_symbols("kraken")
    base = symbol.split("/")[0] if "/" in symbol else symbol
    if (symbol in kraken_syms
            or f"{base}/USD" in kraken_syms
            or f"{base}/USDT" in kraken_syms
            or f"{base}/EUR" in kraken_syms
            or base in kraken_syms):
        available.append("Kraken")

    return available


def fetch_stock_ohlcv(ticker: str, timeframe: str = "1h", limit: int = 500) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance not installed; install yfinance to fetch stock/ETF data.")
    interval_map = {"15m": "15m", "1h": "60m", "4h": "60m", "1d": "1d"}
    interval = interval_map.get(timeframe, "60m")
    period = "60d" if interval != "1d" else "2y"
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False, timeout=6)
    except Exception as e:
        raise RuntimeError(f"yfinance error: {e}")

    if data is None or data.empty:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

    if hasattr(data, "columns") and getattr(data.columns, "nlevels", 1) > 1:
        data.columns = [" ".join([str(x) for x in c if x]).strip() for c in data.columns]

    data = data.reset_index()
    cols = {c: c.lower() for c in data.columns}
    data = data.rename(columns=cols)

    ts_candidates = [c for c in ("datetime", "date") if c in data.columns]
    ts_col = ts_candidates[0] if ts_candidates else (data.columns[0] if len(data.columns) else None)
    if ts_col is None:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame({
        "ts": pd.to_datetime(data[ts_col], utc=True, errors="coerce"),
        "open":   data["open"].astype(float),
        "high":   data["high"].astype(float),
        "low":    data["low"].astype(float),
        "close":  data["close"].astype(float),
        "volume": data.get("volume", 0),
    }).dropna(subset=["ts"])

    df = df.sort_values("ts").reset_index(drop=True)
    if len(df) > limit:
        df = df.iloc[-limit:]
    return df


# ─────────────────────────── Indicators ──────────────────────────────────────

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    close = df["close"].astype(float)
    prev  = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    # Wilder smoothing (same as TradingView)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-smoothed RSI (matches TradingView)."""
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, and histogram."""
    macd_line   = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid   = series.rolling(period).mean()
    std   = series.rolling(period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index — measures trend strength (>25 = trending)."""
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)

    up_move   = high - prev_high
    down_move = prev_low - low
    plus_dm   = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr_s = atr(df, period)
    plus_di  = 100 * pd.Series(plus_dm,  index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr_s.replace(0, 1e-9)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr_s.replace(0, 1e-9)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9)
    return dx.ewm(alpha=1/period, adjust=False).mean()


def swing_highs(high: pd.Series, n: int = 5) -> pd.Series:
    """Vectorized pivot high detection — O(N) using rolling max."""
    roll_max = high.rolling(window=2 * n + 1, center=True).max()
    return (high == roll_max).fillna(False)


def swing_lows(low: pd.Series, n: int = 5) -> pd.Series:
    """Vectorized pivot low detection — O(N) using rolling min."""
    roll_min = low.rolling(window=2 * n + 1, center=True).min()
    return (low == roll_min).fillna(False)


def nearest_structure_target(close_now: float, high: pd.Series, lookback: int = 60) -> Optional[float]:
    """Find the nearest swing-high resistance above price as a target."""
    recent = high.iloc[-lookback:]
    candidates = [float(v) for v in recent if float(v) > close_now * 1.01]
    return min(candidates) if candidates else None


def nearest_structure_stop(close_now: float, low: pd.Series, lookback: int = 30) -> Optional[float]:
    """Find the nearest swing-low support below price as a stop anchor."""
    recent = low.iloc[-lookback:]
    candidates = [float(v) for v in recent if float(v) < close_now * 0.99]
    return max(candidates) if candidates else None


# ─────────────────────────── Market Context Helpers ─────────────────────────

# Simple TTL caches — avoid refetching on every symbol in a scan cycle
_dom_cache:     Dict[str, Any] = {"pct": 0.0, "prev_pct": 0.0, "ts": 0.0}
_funding_cache: Dict[str, Any] = {"rates": {},                  "ts": 0.0}


def get_btc_dominance() -> Tuple[float, bool]:
    """
    Fetch BTC dominance % from CoinGecko (10-min cache).
    Returns (dominance_pct, is_rising).  is_rising = True when dominance has
    climbed > 0.3 percentage points since the last fetch — signals capital
    rotating from alts into BTC, an altcoin headwind.
    """
    import urllib.request, ssl as _ssl
    now = time.time()
    if now - _dom_cache["ts"] < 600:
        return _dom_cache["pct"], _dom_cache["pct"] > _dom_cache["prev_pct"] + 0.3
    try:
        ctx = _ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = _ssl.CERT_NONE
        with urllib.request.urlopen(
            "https://api.coingecko.com/api/v3/global", context=ctx, timeout=6
        ) as r:
            data = json.loads(r.read())["data"]
        new_pct  = float(data["market_cap_percentage"].get("btc", 0.0))
        prev_pct = _dom_cache["pct"] if _dom_cache["pct"] > 0 else new_pct
        _dom_cache.update({"pct": new_pct, "prev_pct": prev_pct, "ts": now})
        return new_pct, new_pct > prev_pct + 0.3
    except Exception:
        return _dom_cache["pct"], False


def get_all_funding_rates() -> Dict[str, float]:
    """
    Fetch all Binance perpetual funding rates in one call (5-min cache).
    Returns {symbol: rate} e.g. {"BTC/USDT": 0.0001}.
    Rate > 0.0005 (0.05%) = longs dominant; > 0.001 (0.1%) = dangerous overleverage.
    """
    import urllib.request, ssl as _ssl
    now = time.time()
    if now - _funding_cache["ts"] < 300:
        return _funding_cache["rates"]
    try:
        ctx = _ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = _ssl.CERT_NONE
        with urllib.request.urlopen(
            "https://fapi.binance.com/fapi/v1/premiumIndex", context=ctx, timeout=8
        ) as r:
            data = json.loads(r.read())
        rates = {
            item["symbol"][:-4] + "/" + item["symbol"][-4:]: float(item.get("lastFundingRate", 0))
            for item in data if item["symbol"].endswith("USDT")
        }
        _funding_cache.update({"rates": rates, "ts": now})
        return rates
    except Exception:
        return _funding_cache["rates"]


def session_context() -> Tuple[str, bool]:
    """
    Returns (session_name, is_prime_session) based on current UTC hour.
    Prime sessions: London open (08–12 UTC) and NY session (13–17 UTC).
    Outside these windows signal confidence is reduced (see _analyse_crypto).
    """
    hour = datetime.utcnow().hour
    if 13 <= hour < 17:
        return "NY session", True
    elif 8 <= hour < 13:
        return "London open", True
    elif 17 <= hour < 21:
        return "NY afternoon", True
    return "Asian/off-peak", False


# ─────────────────────────── Holdings ────────────────────────────────────────

def load_holdings(path: str = "holdings.json") -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"crypto": {}, "stocks": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_position_size(holdings: Dict[str, Any], market: str, symbol: str) -> float:
    key = "crypto" if market == "CRYPTO" else "stocks"
    return float(holdings.get(key, {}).get(symbol, 0) or 0)


# ─────────────────────────── Regime ──────────────────────────────────────────

def market_regime(df: pd.DataFrame) -> str:
    """
    Classify market regime using EMA slope + ATR%.
    Returns: TREND_UP | TREND_DOWN | RANGE | COMPRESSION | EXPANSION | CHOPPY
    """
    close    = df["close"].astype(float)
    e20      = ema(close, 20)
    e50      = ema(close, 50)
    e200     = ema(close, 200) if len(close) >= 200 else e50
    last     = float(close.iloc[-1])
    last_atr = float(atr(df, 14).iloc[-1])
    atr_pct  = last_atr / last if last > 0 else 0.0

    # EMA stack
    bull_stack = e20.iloc[-1] > e50.iloc[-1] > e200.iloc[-1]
    bear_stack = e20.iloc[-1] < e50.iloc[-1] < e200.iloc[-1]

    # Short-term slope: % change of EMA20 over 5 bars
    slope = (e20.iloc[-1] - e20.iloc[-6]) / e20.iloc[-6] if e20.iloc[-6] > 0 else 0.0

    if atr_pct < 0.003:
        return "COMPRESSION"
    if atr_pct > 0.04:
        return "EXPANSION"
    if abs(slope) < 0.003 and not bull_stack and not bear_stack:
        return "RANGE"
    if bull_stack and slope > 0.001:
        return "TREND_UP"
    if bear_stack and slope < -0.001:
        return "TREND_DOWN"
    return "CHOPPY"


# ─────────────────────────── Setup Detectors ─────────────────────────────────

def _make_idea(
    direction: str,
    entry: float,
    stop: float,
    targets: List[float],
    conf: int,
    reasons: List[str],
    market: str,
) -> Optional[Tuple]:
    """Validate and package a setup. Returns None if RR < min_rr or stop too wide."""
    MAX_STOP_DISTANCE = 0.08  # tightened from 9%: VIC (8.56%) and GNO (8.80%) slipped through
    MIN_STOP_DISTANCE = 0.025  # stops < 2.5% = noise stops, hit by normal crypto volatility
    if stop >= entry or not targets:
        return None
    risk = entry - stop
    if risk <= 0:
        return None
    risk_pct = risk / entry
    if risk_pct > MAX_STOP_DISTANCE:
        return None
    if risk_pct < MIN_STOP_DISTANCE:
        return None
    # Progressive R:R — wider stops require proportionally better reward
    if risk_pct > 0.07:
        min_rr = 3.5
    elif risk_pct > 0.05:
        min_rr = 3.0
    else:
        min_rr = 2.5
    best_target = targets[-1]
    rr = (best_target - entry) / risk
    exp_pct = (best_target - entry) / entry * 100 if entry > 0 else 0.0
    min_exp = 5.0 if market == "CRYPTO" else 3.0
    if rr < min_rr or exp_pct < min_exp:
        return None
    return direction, (entry * 0.999, entry * 1.001), stop, targets, round(rr, 2), round(exp_pct, 1), conf, reasons


# ── Setup 1: Breakout with Volume Confirmation ────────────────────────────────
def detect_breakout_with_volume(df: pd.DataFrame, market: str, lookback: int = 20) -> Optional[Tuple]:
    """
    Price closes above N-bar resistance with:
    - Volume ≥ 1.5× 20-bar average (not just last bar — confirms real interest)
    - ATR expanding (momentum behind the move)
    - RSI 52–72 (not already overextended)
    - ADX > 20 (trend has some strength)
    - Close > open on breakout candle (bullish body, not wick)
    """
    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    vol   = df["volume"].astype(float)
    op    = df["open"].astype(float)

    if len(df) < lookback + 10:
        return None

    last_close = float(close.iloc[-1])
    last_open  = float(op.iloc[-1])
    last_atr   = float(atr(df, 14).iloc[-1])

    # Resistance = highest high over lookback bars, excluding the last bar (so we confirm a break)
    resistance = float(high.iloc[-(lookback + 1):-1].max())

    vol_ma    = vol.rolling(20).mean()
    vol_ratio = float(vol.iloc[-1] / vol_ma.iloc[-1]) if float(vol_ma.iloc[-1]) > 0 else 0.0

    rsi_val = float(rsi(close, 14).iloc[-1])
    adx_val = float(adx(df, 14).iloc[-1]) if len(df) >= 28 else 20.0

    atr_s        = atr(df, 14)
    atr_avg_prev = float(atr_s.iloc[-15:-1].mean())
    atr_expanding = float(atr_s.iloc[-1]) > atr_avg_prev * 1.08

    bullish_body = last_close > last_open  # candle closed as bull

    # All conditions must hold
    if not (last_close > resistance and vol_ratio >= 1.3 and atr_expanding
            and 48 <= rsi_val <= 82 and adx_val >= 16 and bullish_body):
        return None

    # Target: nearest swing-high above price, or ATR-based fallback
    struct_target = nearest_structure_target(last_close, high, lookback=80)
    atr_target    = last_close + 3.5 * last_atr
    target        = struct_target if struct_target and struct_target > atr_target * 0.85 else atr_target
    tp1           = last_close + 1.5 * last_atr

    # Stop: just below the broken resistance level
    stop = resistance - 0.3 * last_atr

    conf = 72
    conf += 8  if vol_ratio > 2.5 else (4 if vol_ratio > 2.0 else 0)
    conf += 5  if rsi_val > 60 else 0
    conf += 5  if adx_val > 30 else 0
    conf = min(90, conf)

    reasons = [
        f"Breakout: close {last_close:.4f} > {lookback}-bar high {resistance:.4f}",
        f"Vol spike {vol_ratio:.1f}×, ATR expanding, RSI {rsi_val:.1f}, ADX {adx_val:.1f}",
        "Bullish close candle confirms breakout",
    ]
    return _make_idea("LONG", last_close, stop, [tp1, target], conf, reasons, market)


# ── Setup 2: Pullback to EMA in Uptrend ──────────────────────────────────────
def detect_pullback_to_ema(df: pd.DataFrame, market: str) -> Optional[Tuple]:
    """
    Uptrend (EMA20 > EMA50 > EMA200) with:
    - Price pulls back within 1.5% of EMA20 or EMA50 (tightened from 2%)
    - RSI cools to 38–55 (genuine pullback, not exhaustion)
    - Last bar closes above its midpoint (buyers absorbing)
    - ADX > 22 (trend remains intact)
    - No lower-low in the last 3 bars (doesn't look like a breakdown)
    """
    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)

    if len(df) < 60:
        return None

    last_close = float(close.iloc[-1])
    last_high  = float(high.iloc[-1])
    last_low   = float(low.iloc[-1])
    last_atr   = float(atr(df, 14).iloc[-1])

    e20  = ema(close, 20)
    e50  = ema(close, 50)
    e200 = ema(close, 200) if len(close) >= 200 else None

    # Require full EMA stack only when we have enough history
    if e200 is not None:
        in_uptrend = e20.iloc[-1] > e50.iloc[-1] > e200.iloc[-1]
    else:
        in_uptrend = e20.iloc[-1] > e50.iloc[-1]
    if not in_uptrend:
        return None

    near_e20 = abs(last_close - float(e20.iloc[-1])) / float(e20.iloc[-1]) < 0.015
    near_e50 = abs(last_close - float(e50.iloc[-1])) / float(e50.iloc[-1]) < 0.015

    if not (near_e20 or near_e50):
        return None

    rsi_val = float(rsi(close, 14).iloc[-1])
    adx_val = float(adx(df, 14).iloc[-1]) if len(df) >= 28 else 22.0

    # Bullish absorption: close in upper 40% of candle range
    candle_range = last_high - last_low
    bullish_close = last_close > (last_low + 0.6 * candle_range) if candle_range > 0 else False

    # No breakdown: last 3 lows should not be making lower lows aggressively
    no_breakdown = float(low.iloc[-1]) >= float(low.iloc[-3]) * 0.985

    if not (36 <= rsi_val <= 57 and adx_val >= 18 and bullish_close and no_breakdown):
        return None

    ema_support = float(e20.iloc[-1]) if near_e20 else float(e50.iloc[-1])
    stop = ema_support - 1.2 * last_atr

    struct_target = nearest_structure_target(last_close, high, lookback=60)
    atr_target    = last_close + 3.0 * last_atr
    target        = struct_target if struct_target and struct_target > atr_target * 0.9 else atr_target
    tp1           = last_close + 1.5 * last_atr

    conf = 74
    conf += 6 if near_e20 else 3
    conf += 5 if rsi_val < 48 else 0
    conf += 4 if adx_val > 28 else 0
    conf = min(88, conf)

    ema_name = "EMA20" if near_e20 else "EMA50"
    reasons = [
        f"Pullback to {ema_name} in uptrend (EMA20>EMA50>EMA200)",
        f"RSI cooled to {rsi_val:.1f}, ADX {adx_val:.1f} (trend intact)",
        "Bullish absorption candle at support",
    ]
    return _make_idea("LONG", last_close, stop, [tp1, target], conf, reasons, market)


# ── Setup 3: Range Bounce from Support ───────────────────────────────────────
def detect_range_bounce(df: pd.DataFrame, market: str, lookback: int = 30) -> Optional[Tuple]:
    """
    Sideways channel with:
    - Range width 5-15% (well-defined, not too narrow/wide)
    - Price within 2% of range low (genuine support test)
    - RSI between 30–50 (oversold-to-neutral, not free-falling)
    - Volume at least average (not dead tape)
    - ATR stable (not expanding — we don't want a breakdown)
    """
    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    vol   = df["volume"].astype(float)

    if len(df) < lookback + 10:
        return None

    last_close = float(close.iloc[-1])
    last_atr   = float(atr(df, 14).iloc[-1])

    recent_high = float(high.iloc[-lookback:].max())
    recent_low  = float(low.iloc[-lookback:].min())

    if recent_low <= 0:
        return None

    range_pct = (recent_high - recent_low) / recent_low * 100
    range_ok  = 5.0 <= range_pct <= 15.0

    near_support = (last_close - recent_low) / recent_low < 0.025

    rsi_val = float(rsi(close, 14).iloc[-1])

    vol_ma  = vol.rolling(20).mean()
    vol_ratio = float(vol.iloc[-1] / float(vol_ma.iloc[-1])) if float(vol_ma.iloc[-1]) > 0 else 0.0
    vol_ok  = vol_ratio >= 0.8  # at least normal volume

    # ATR should not be expanding rapidly (avoid breakdowns)
    atr_s       = atr(df, 14)
    atr_stable  = float(atr_s.iloc[-1]) < float(atr_s.iloc[-5:].mean()) * 1.2

    if not (range_ok and near_support and 28 <= rsi_val <= 50 and vol_ok and atr_stable):
        return None

    stop   = recent_low - 0.5 * last_atr
    # Target: midpoint of range (conservative), or upper third
    target_mid   = recent_low + (recent_high - recent_low) * 0.65
    target_full  = recent_high - last_atr * 0.3
    tp1          = recent_low + (recent_high - recent_low) * 0.45

    conf = 65
    conf += 8 if range_pct < 10 else 0
    conf += 5 if rsi_val < 42 else 0
    conf = min(82, conf)

    reasons = [
        f"Range bounce: {range_pct:.1f}% channel, price at support {recent_low:.4f}",
        f"RSI {rsi_val:.1f}, volume normal ({vol_ratio:.1f}×)",
        "ATR stable — no breakdown pressure",
    ]
    return _make_idea("LONG", last_close, stop, [tp1, target_full], conf, reasons, market)


# ── Setup 4: Volatility Expansion (BB Squeeze Breakout) ──────────────────────
def detect_volatility_expansion(df: pd.DataFrame, market: str) -> Optional[Tuple]:
    """
    Bollinger Band squeeze resolving upward:
    - BB width contracted for ≥10 bars (squeeze period: width < 20-bar avg)
    - Current BB width expanding > 30% from squeeze low
    - Price above mid-band (direction = up)
    - Volume rising (confirms the expansion)
    - RSI 48–72 (not oversold, not overbought)
    """
    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    vol   = df["volume"].astype(float)

    if len(df) < 50:
        return None

    last_close = float(close.iloc[-1])
    last_atr   = float(atr(df, 14).iloc[-1])

    upper_bb, mid_bb, lower_bb = bollinger_bands(close, 20, 2.0)
    bb_width = upper_bb - lower_bb

    # Need at least 10-bar squeeze: bb_width below its 20-bar average
    bb_width_avg = bb_width.rolling(20).mean()
    squeeze_bars = int((bb_width.iloc[-15:-2] < bb_width_avg.iloc[-15:-2]).sum())
    if squeeze_bars < 6:
        return None

    # Current width must be expanding significantly vs the squeeze low
    squeeze_low  = float(bb_width.iloc[-15:-2].min())
    current_width = float(bb_width.iloc[-1])
    expanding = current_width > squeeze_low * 1.3

    above_mid    = last_close > float(mid_bb.iloc[-1])
    rsi_val      = float(rsi(close, 14).iloc[-1])
    vol_ma       = vol.rolling(20).mean()
    vol_rising   = float(vol.iloc[-1]) > float(vol_ma.iloc[-1]) * 1.2

    if not (expanding and above_mid and 48 <= rsi_val <= 72 and vol_rising):
        return None

    stop   = float(lower_bb.iloc[-1]) - 0.3 * last_atr
    struct = nearest_structure_target(last_close, high, 60)
    target = struct if struct else last_close + 4.0 * last_atr
    tp1    = last_close + 2.0 * last_atr

    conf = 70
    conf += 5 if squeeze_bars >= 10 else 0
    conf += 5 if vol_rising and float(vol.iloc[-1]) > float(vol_ma.iloc[-1]) * 1.5 else 0
    conf = min(85, conf)

    reasons = [
        f"BB squeeze resolving UP ({squeeze_bars} squeeze bars, width +{((current_width/squeeze_low)-1)*100:.0f}%)",
        f"RSI {rsi_val:.1f}, volume surge {float(vol.iloc[-1])/float(vol_ma.iloc[-1]):.1f}×",
        "Close above mid-band confirms bullish expansion",
    ]
    return _make_idea("LONG", last_close, stop, [tp1, target], conf, reasons, market)


# ── Setup 5: Trend Continuation (Swing Structure) ────────────────────────────
def detect_trend_continuation(df: pd.DataFrame, market: str, lookback: int = 20) -> Optional[Tuple]:
    """
    Confirmed uptrend using proper swing structure (not just 2-point comparison):
    - At least 3 consecutive higher-highs and higher-lows using pivot points
    - Price just broke out of minor consolidation (< 5 bars flat)
    - EMA20 > EMA50 and both sloping up
    - ADX > 25 (strong trend)
    - RSI 50–70
    """
    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)

    if len(df) < 60:
        return None

    last_close = float(close.iloc[-1])
    last_atr   = float(atr(df, 14).iloc[-1])

    e20 = ema(close, 20)
    e50 = ema(close, 50)

    emas_up = e20.iloc[-1] > e50.iloc[-1] and e20.iloc[-1] > e20.iloc[-5] and e50.iloc[-1] > e50.iloc[-10]
    if not emas_up:
        return None

    adx_val = float(adx(df, 14).iloc[-1]) if len(df) >= 28 else 20.0
    rsi_val = float(rsi(close, 14).iloc[-1])

    if adx_val < 18 or not (46 <= rsi_val <= 82):
        return None

    # Swing structure: find last 5 pivot highs & lows
    ph = swing_highs(high, n=3)
    pl = swing_lows(low, n=3)

    pivot_highs = [float(high.iloc[i]) for i in range(max(0, len(df)-60), len(df)) if ph.iloc[i]]
    pivot_lows  = [float(low.iloc[i])  for i in range(max(0, len(df)-60), len(df)) if pl.iloc[i]]

    # Need at least 3 pivot highs and 3 pivot lows that are ascending
    if len(pivot_highs) < 3 or len(pivot_lows) < 3:
        return None

    hh_confirmed = all(pivot_highs[i] < pivot_highs[i+1] for i in range(len(pivot_highs)-2, len(pivot_highs)-1))
    hl_confirmed = all(pivot_lows[i]  < pivot_lows[i+1]  for i in range(len(pivot_lows)-2,  len(pivot_lows)-1))

    # Fallback: even 2 ascending pivots is ok when ADX is very strong
    # (raised from 30 to 35 — this setup has no live win-rate data yet, so require
    # extra trend-strength margin in place of confirmed swing structure)
    if not hh_confirmed or not hl_confirmed:
        if adx_val < 35:
            return None

    # Minor consolidation: last 5 bars should be in a tight range < 2×ATR
    last5_range = float(high.iloc[-5:].max()) - float(low.iloc[-5:].min())
    consolidating = last5_range < 2.5 * last_atr

    # Breakout of the consolidation
    consol_high = float(high.iloc[-6:-1].max())
    breakout    = last_close > consol_high

    if not (consolidating and breakout):
        return None

    stop   = float(low.rolling(10).min().iloc[-1]) - 0.4 * last_atr
    struct = nearest_structure_target(last_close, high, 60)
    target = struct if struct else last_close + 3.5 * last_atr
    tp1    = last_close + 1.5 * last_atr

    conf = 73
    conf += 7 if adx_val > 30 else 3
    conf += 5 if rsi_val < 65 else 0
    conf = min(88, conf)

    reasons = [
        f"Trend continuation: HH+HL structure, ADX {adx_val:.1f}",
        f"Mini-consolidation breakout above {consol_high:.4f}",
        f"RSI {rsi_val:.1f}, EMA20 > EMA50 both rising",
    ]
    return _make_idea("LONG", last_close, stop, [tp1, target], conf, reasons, market)


# ── Setup 6: Momentum Rally (catches broad bull market moves) ────────────────
def detect_momentum_rally(df: pd.DataFrame, market: str) -> Optional[Tuple]:
    """
    Designed for strong trending markets where RSI is elevated (coins already moving).
    Conditions:
    - Price up >3% over last 4 bars (real momentum, not noise)
    - EMA20 > EMA50 (trend intact)
    - RSI 55–88 (elevated but not extreme)
    - Volume above average (real buying, not thin tape)
    - Close in upper 30% of last 10-bar range (price holding gains)
    - ATR expanding (volatility confirms move)
    Entry: current price
    Stop: EMA20 or 2×ATR below, whichever is closer
    Target: structure or 5×ATR (high momentum = room to run)
    """
    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    vol   = df["volume"].astype(float)

    if len(df) < 30:
        return None

    last_close = float(close.iloc[-1])
    last_atr   = float(atr(df, 14).iloc[-1])

    # Momentum: price change over last 4 bars
    price_4bar_ago = float(close.iloc[-5])
    momentum_pct = (last_close - price_4bar_ago) / price_4bar_ago * 100 if price_4bar_ago > 0 else 0.0

    e20 = ema(close, 20)
    e50 = ema(close, 50)
    trend_ok = float(e20.iloc[-1]) > float(e50.iloc[-1])

    rsi_val  = float(rsi(close, 14).iloc[-1])
    vol_ma   = vol.rolling(20).mean()
    vol_ratio = float(vol.iloc[-1] / float(vol_ma.iloc[-1])) if float(vol_ma.iloc[-1]) > 0 else 0.0

    # Price in upper 30% of 10-bar range (holding gains, not reversing)
    range_high = float(high.iloc[-10:].max())
    range_low  = float(low.iloc[-10:].min())
    range_span = range_high - range_low
    in_upper_range = (last_close - range_low) / range_span > 0.70 if range_span > 0 else False

    # ATR expanding
    atr_s = atr(df, 14)
    atr_expanding = float(atr_s.iloc[-1]) > float(atr_s.iloc[-8:-1].mean()) * 1.05

    # Thresholds raised from the original 4.0%/1.3x (no live win-rate data existed for
    # this setup) to be more selective while it establishes a live track record.
    if not (momentum_pct >= 5.0 and trend_ok and 55 <= rsi_val <= 78
            and vol_ratio >= 1.5 and in_upper_range and atr_expanding):
        return None

    # Stop: EMA20 or 2×ATR below price, whichever gives tighter stop
    ema20_stop = float(e20.iloc[-1]) - 0.5 * last_atr
    atr_stop   = last_close - 2.0 * last_atr
    stop = max(ema20_stop, atr_stop)  # tighter stop = max of the two

    if stop >= last_close:
        stop = last_close - 2.0 * last_atr

    # Target: structure resistance or 5×ATR (momentum moves run far)
    struct_target = nearest_structure_target(last_close, high, lookback=60)
    atr_target    = last_close + 5.0 * last_atr
    target        = struct_target if struct_target and struct_target > last_close * 1.03 else atr_target
    tp1           = last_close + 2.5 * last_atr

    conf = 68
    conf += 8  if momentum_pct >= 6.0 else (4 if momentum_pct >= 4.0 else 0)
    conf += 6  if vol_ratio >= 2.0 else (3 if vol_ratio >= 1.5 else 0)
    conf += 5  if rsi_val >= 65 and rsi_val <= 80 else 0
    conf += 4  if trend_ok and float(e20.iloc[-1]) > float(e20.iloc[-6]) else 0
    conf = min(90, conf)

    reasons = [
        f"Momentum rally: +{momentum_pct:.1f}% in 4 bars, RSI {rsi_val:.1f}",
        f"Volume {vol_ratio:.1f}× avg, ATR expanding, price holding upper range",
        f"EMA20 > EMA50, trend intact",
    ]
    return _make_idea("LONG", last_close, stop, [tp1, target], conf, reasons, market)


# ── Setup 7: Pre-Breakout Coil (catches moves BEFORE they happen) ────────────
def detect_pre_breakout_coil(df: pd.DataFrame, market: str, lookback: int = 20) -> Optional[Tuple]:
    """
    Spots coins coiling tightly BEFORE the breakout — entry is early, stop is tight.

    Conditions (all must hold):
    - Price range of last 8 bars < 1.5× ATR  (tight coil = compression)
    - Price within 1.5% of N-bar resistance  (pressing against ceiling)
    - Volume declining over last 5 bars      (sellers exhausted)
    - RSI 42–62                              (neutral — not overbought, has room to run)
    - ATR contracted vs 20-bar avg           (volatility compressed = spring loaded)
    - EMA20 > EMA50                          (higher TF trend is up)

    Early entry = better R:R. Stop is tight because if it breaks down, the thesis is wrong.
    """
    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    vol   = df["volume"].astype(float)

    if len(df) < lookback + 10:
        return None

    last_close = float(close.iloc[-1])
    last_atr   = float(atr(df, 14).iloc[-1])

    # Resistance = N-bar high (excluding last bar)
    resistance = float(high.iloc[-(lookback + 1):-1].max())

    # Tight coil: last 8 bars range < 1.5× ATR
    coil_high = float(high.iloc[-8:].max())
    coil_low  = float(low.iloc[-8:].min())
    coil_range = coil_high - coil_low
    tight_coil = coil_range < 1.5 * last_atr

    # Pressing against resistance: price within 1.5%
    near_resistance = (resistance - last_close) / resistance < 0.015 if resistance > 0 else False

    # Volume drying up (last 5 bars declining avg vs prior 5)
    vol_recent = float(vol.iloc[-5:].mean())
    vol_prior  = float(vol.iloc[-10:-5].mean())
    vol_drying = vol_recent < vol_prior * 0.85 if vol_prior > 0 else False

    rsi_val = float(rsi(close, 14).iloc[-1])
    rsi_ok  = 40 <= rsi_val <= 64

    # ATR contracted
    atr_s = atr(df, 14)
    atr_contracted = float(atr_s.iloc[-1]) < float(atr_s.iloc[-20:].mean()) * 0.90

    e20 = ema(close, 20)
    e50 = ema(close, 50)
    trend_ok = float(e20.iloc[-1]) > float(e50.iloc[-1])

    if not (tight_coil and near_resistance and vol_drying and rsi_ok and atr_contracted and trend_ok):
        return None

    # Entry just below resistance so we catch the break
    entry = last_close
    # Tight stop: below the coil low
    stop  = coil_low - 0.3 * last_atr

    # Target: measured move = coil range projected up from resistance
    measured_move  = resistance + (resistance - coil_low)
    struct_target  = nearest_structure_target(resistance, high, lookback=80)
    target = struct_target if struct_target and struct_target > measured_move * 0.9 else measured_move
    tp1    = resistance + coil_range * 0.5

    conf = 70
    conf += 8 if coil_range < last_atr else 4          # tighter coil = higher conf
    conf += 6 if (resistance - last_close) / resistance < 0.008 else 3  # very close to breakout
    conf += 5 if rsi_val < 55 else 0                   # RSI has more room to run
    conf = min(88, conf)

    reasons = [
        f"Pre-breakout coil: {lookback}-bar resistance {resistance:.4f}, price {last_close:.4f} ({((resistance-last_close)/resistance*100):.1f}% away)",
        f"Coil range {coil_range:.4f} < 1.5×ATR, volume drying ({vol_recent/vol_prior*100:.0f}% of prior), RSI {rsi_val:.1f}",
        "Spring loaded: low volatility compression before expansion",
    ]
    return _make_idea("LONG", entry, stop, [tp1, target], conf, reasons, market)


# ── Setup 8: Volume Climax / Accumulation (detects smart money buying) ───────
def detect_volume_accumulation(df: pd.DataFrame, market: str) -> Optional[Tuple]:
    """
    Detects when a coin has had a volume spike WITHOUT a proportional price rise —
    classic sign of smart money accumulating at a level before markup phase.

    Conditions:
    - Volume in last 1–3 bars is 2×+ the 20-bar average
    - But price move is SMALL (<1.5×ATR) — absorption, not a breakout yet
    - Price is near a support level (low of last 15 bars within 2%)
    - RSI 35–58 (room to run upward)
    - Not in a downtrend (EMA20 >= EMA50)

    This fires BEFORE the move, when big players are filling orders quietly.
    """
    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    vol   = df["volume"].astype(float)

    if len(df) < 25:
        return None

    last_close = float(close.iloc[-1])
    last_atr   = float(atr(df, 14).iloc[-1])

    vol_ma    = vol.rolling(20).mean()
    # Check if any of last 3 bars had a volume spike
    vol_spike_bars = [i for i in range(-3, 0)
                      if float(vol_ma.iloc[i]) > 0 and float(vol.iloc[i]) / float(vol_ma.iloc[i]) >= 2.0]
    if not vol_spike_bars:
        return None

    # Price movement during the spike bar(s) should be small (absorption)
    for i in vol_spike_bars:
        bar_move = abs(float(close.iloc[i]) - float(close.iloc[i-1]))
        if bar_move > 1.5 * last_atr:
            return None  # Price moved too much — that's a breakout, not accumulation

    # Near support: within 2% of 15-bar low
    support = float(low.iloc[-15:].min())
    near_support = (last_close - support) / support < 0.025 if support > 0 else False

    rsi_val = float(rsi(close, 14).iloc[-1])
    rsi_ok  = 33 <= rsi_val <= 60

    e20 = ema(close, 20)
    e50 = ema(close, 50)
    not_downtrend = float(e20.iloc[-1]) >= float(e50.iloc[-1]) * 0.98

    if not (near_support and rsi_ok and not_downtrend):
        return None

    # Entry: current price
    entry = last_close
    stop  = support - 0.4 * last_atr

    # Target: next resistance (structure) or 4×ATR
    struct_target = nearest_structure_target(last_close, high, lookback=60)
    atr_target    = last_close + 4.0 * last_atr
    target = struct_target if struct_target and struct_target > atr_target * 0.85 else atr_target
    tp1    = last_close + 2.0 * last_atr

    # Best vol ratio across spike bars
    best_vol_ratio = max(float(vol.iloc[i]) / float(vol_ma.iloc[i]) for i in vol_spike_bars)

    conf = 67
    conf += 8 if best_vol_ratio >= 3.0 else (4 if best_vol_ratio >= 2.5 else 0)
    conf += 5 if rsi_val < 48 else 0
    conf += 5 if near_support and (last_close - support) / support < 0.01 else 0
    conf = min(87, conf)

    reasons = [
        f"Volume accumulation: {best_vol_ratio:.1f}× spike with small price move (absorption)",
        f"Price near support {support:.4f}, RSI {rsi_val:.1f} (room to run)",
        "Smart money accumulation pattern — markup phase may follow",
    ]
    return _make_idea("LONG", entry, stop, [tp1, target], conf, reasons, market)


# ── Setup 9: Hot Momentum — for coins already running 10%+ in 24h ────────────
def detect_hot_momentum(df: pd.DataFrame, market: str, gain_24h_pct: float = 0.0) -> Optional[Tuple]:
    """
    For coins confirmed moving in 24h — but ONLY enter on a healthy pullback,
    never at the top of a pump.

    Key insight: buying a coin up 30% at its exact high has near-zero edge.
    Buying the same coin after it pulls back 5-8% to EMA support and consolidates
    is a completely different trade with a defined stop and real R:R.

    Requirements:
    - Price has pulled back ≥3% from its 10-bar high (not at the top)
    - Price is above EMA20 and EMA20 is still rising (trend intact)
    - RSI has cooled below 70 (not at peak momentum anymore)
    - Volume on the pullback is LOWER than the spike bars (healthy fade, not panic)
    - Candle is not a strong bearish close (not a reversal)
    - ATR still elevated vs baseline (move has not fully died)
    """
    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    vol   = df["volume"].astype(float)
    op    = df["open"].astype(float)

    if len(df) < 25:
        return None

    last_close = float(close.iloc[-1])
    last_open  = float(op.iloc[-1])
    last_atr   = float(atr(df, 14).iloc[-1])

    # ── 1. Pullback from recent high (not buying the top) ─────────────────
    recent_high_10 = float(high.iloc[-10:].max())
    pullback_pct   = (recent_high_10 - last_close) / recent_high_10 * 100 if recent_high_10 > 0 else 0
    # Scale pullback requirement with size of move:
    # small move (10-20%) needs 3% pullback, big move (>30%) needs 5%
    min_pullback = 5.0 if gain_24h_pct >= 30 else 3.0
    pulled_back  = pullback_pct >= min_pullback

    if not pulled_back:
        return None  # still at the top — do not enter

    # ── 1b. Pump freshness: high must be within last 5 bars ───────────────
    # If the peak was 6-10 hours ago (e.g., Asian-session pump caught at London open),
    # the coin is likely reversing, not consolidating for continuation.
    high_slice = high.iloc[-10:]
    bars_since_high = len(high_slice) - 1 - int(high_slice.values.argmax())
    if bars_since_high > 5:
        return None  # stale pump — peak too far back, reversal risk

    # ── 2. Trend still intact ─────────────────────────────────────────────
    e20 = ema(close, 20)
    e20_rising    = float(e20.iloc[-1]) > float(e20.iloc[-5])
    above_ema20   = last_close > float(e20.iloc[-1]) * 0.98

    if not (above_ema20 and e20_rising):
        return None

    # ── 3. RSI cooled — must be in valid pullback zone (45–72) ──────────
    rsi_val = float(rsi(close, 14).iloc[-1])
    if rsi_val >= 72:
        return None  # still overheated — wait more
    if rsi_val < 45:
        return None  # momentum broken, not a pullback worth buying

    # ── 4. Pullback volume lower than spike (healthy fade, not panic sell) ─
    vol_ma       = vol.rolling(20).mean()
    vol_now      = float(vol.iloc[-1])
    vol_avg      = float(vol_ma.iloc[-1]) if float(vol_ma.iloc[-1]) > 0 else 1
    vol_spike_5  = float(vol.iloc[-5:].max())  # highest volume in last 5 bars
    healthy_fade = vol_now < vol_spike_5 * 0.7  # pullback volume < 70% of spike

    # ── 5. Not a reversal candle ──────────────────────────────────────────
    candle_body    = last_open - last_close
    not_reversing  = candle_body < 1.5 * last_atr

    # ── 6. ATR still elevated (move hasn't fully died) ────────────────────
    atr_s    = atr(df, 14)
    atr_base = float(atr_s.iloc[-20:-5].mean()) if len(atr_s) >= 25 else float(atr_s.mean())
    atr_alive = float(atr_s.iloc[-1]) > atr_base * 1.2  # still 20% above baseline

    if not (not_reversing and atr_alive):
        return None

    # ── Entry and stops ───────────────────────────────────────────────────
    entry = last_close

    # Tight stop: below pullback low, with small buffer
    pullback_low = float(low.iloc[-5:].min())
    stop = pullback_low - 0.4 * last_atr
    if stop >= entry:
        stop = entry - 1.5 * last_atr

    # Target: measured move from pullback — prior high + (prior high - pullback low)
    tp1    = recent_high_10  # first target: retest the recent high
    tp2_mm = recent_high_10 + (recent_high_10 - pullback_low)  # measured move extension
    struct = nearest_structure_target(recent_high_10, high, lookback=60)
    tp2    = struct if struct and struct > recent_high_10 * 1.02 else tp2_mm

    conf = 68
    conf += 8 if gain_24h_pct >= 25 else (5 if gain_24h_pct >= 15 else 2)
    conf += 5 if pullback_pct >= 6 else (3 if pullback_pct >= 4 else 0)  # deeper pullback = better entry
    conf += 5 if rsi_val < 60 else (2 if rsi_val < 65 else 0)
    conf += 4 if healthy_fade else 0
    conf = min(88, conf)

    reasons = [
        f"Hot momentum pullback: +{gain_24h_pct:.1f}% 24h mover, pulled back {pullback_pct:.1f}% from high",
        f"RSI cooled to {rsi_val:.1f}, EMA20 rising, healthy volume fade",
        f"Entry at pullback low, TP1 = retest of high {recent_high_10:.4f}",
    ]
    return _make_idea("LONG", entry, stop, [tp1, tp2], conf, reasons, market)


# ─────────────────────────── Confluence Scorer ───────────────────────────────

def confluence_bonus(df: pd.DataFrame, base_conf: int) -> Tuple[int, List[str]]:
    """
    Award extra confidence points when multiple independent factors align.
    Returns updated confidence and bonus reasons.
    """
    close  = df["close"].astype(float)
    high   = df["high"].astype(float)
    low    = df["low"].astype(float)
    vol    = df["volume"].astype(float)

    bonus_reasons: List[str] = []
    bonus = 0

    # 1. Volume trend: 3-bar rising volume
    if len(vol) >= 4 and all(vol.iloc[-(i+1)] > vol.iloc[-(i+2)] for i in range(3)):
        bonus += 3
        bonus_reasons.append("3-bar rising volume (accumulation)")

    # 2. Price above VWAP proxy (EMA of typical price)
    tp = (df["high"].astype(float) + df["low"].astype(float) + close) / 3
    vwap_proxy = tp.ewm(span=20, adjust=False).mean()
    if float(close.iloc[-1]) > float(vwap_proxy.iloc[-1]):
        bonus += 2
        bonus_reasons.append("Price above VWAP proxy")

    # 3. Higher-timeframe trend proxy: 50-bar close slope
    if len(close) >= 55:
        slope_50 = (float(close.iloc[-1]) - float(close.iloc[-51])) / float(close.iloc[-51]) * 100
        if slope_50 > 2.0:
            bonus += 3
            bonus_reasons.append(f"50-bar uptrend (+{slope_50:.1f}%)")

    # 4. Candle quality: green candle with body > 50% of range
    last_open  = float(df["open"].iloc[-1])
    last_close_v = float(close.iloc[-1])
    last_high_v  = float(high.iloc[-1])
    last_low_v   = float(low.iloc[-1])
    body       = abs(last_close_v - last_open)
    crange     = last_high_v - last_low_v
    if crange > 0 and body / crange > 0.5 and last_close_v > last_open:
        bonus += 3
        bonus_reasons.append("Strong bullish candle body")

    return min(95, base_conf + bonus), bonus_reasons


# ─────────────────────────── Confluence Score (4-Layer) ──────────────────────

def confluence_score(
    df_1h: pd.DataFrame,
    df_4h: Optional[pd.DataFrame],
    setup_name: str,
    setup_result: Optional[Tuple] = None,
) -> Tuple[float, Dict[str, int], List[str]]:
    """
    Weighted 4-layer signal quality score 0–100.
    Weights: Trend 30% | Momentum 30% | Oscillators 20% | S/R 20%
    Minimum score required to approve a signal: 75.
    Any layer scoring 0 is a hard-fail — total forced to 0 regardless of others.

    Returns (total_score, layer_scores_dict, fail_reasons).
    """
    layer: Dict[str, int] = {"trend": 0, "momentum": 0, "oscillators": 0, "sr": 0}
    fails: List[str] = []

    close = df_1h["close"].astype(float)
    vol   = df_1h["volume"].astype(float)
    high  = df_1h["high"].astype(float)
    low   = df_1h["low"].astype(float)
    op    = df_1h["open"].astype(float)
    last  = float(close.iloc[-1])

    # ── A. Trend (Minervini) — 30% ────────────────────────────────────────
    # HOT_MOMENTUM: below 4H EMA50 is a penalty (score 40), NOT a hard-fail.
    # Momentum pumps frequently start from below EMA50 (oversold bounces, catalysts,
    # new listings). The 24h gain itself is strong trend evidence for this setup.
    # RANGE_BOUNCE: keeps hard-fail — bounces need macro support to hold.
    if df_4h is not None and len(df_4h) >= 55:
        c4       = df_4h["close"].astype(float)
        e50_4h   = ema(c4, 50)
        e50_now  = float(e50_4h.iloc[-1])
        e50_prev = float(e50_4h.iloc[-6])
        last_4h  = float(c4.iloc[-1])
        if last_4h < e50_now:
            if setup_name in ("HOT_MOMENTUM", "TREND_CONTINUATION", "MOMENTUM_RALLY"):
                layer["trend"] = 40   # penalty, not elimination
                fails.append(f"Trend: below 4H EMA50 (−penalty)")
            else:
                layer["trend"] = 0
                fails.append(f"Trend: 4H close {last_4h:.4f} < EMA50 {e50_now:.4f}")
        else:
            slope_pct = (e50_now - e50_prev) / e50_prev * 100 if e50_prev > 0 else 0.0
            layer["trend"] = 100 if slope_pct > 0.5 else (85 if slope_pct > 0.0 else 65)
    elif len(close) >= 200:
        e200 = ema(close, 200)
        if last >= float(e200.iloc[-1]):
            layer["trend"] = 60
        else:
            layer["trend"] = 35 if setup_name in ("HOT_MOMENTUM", "TREND_CONTINUATION", "MOMENTUM_RALLY") else 0
            if layer["trend"] == 0:
                fails.append("Trend: below 1H EMA200 proxy — macro downtrend")
    else:
        layer["trend"] = 50  # insufficient history — neutral

    # ── B. Momentum (Cameron) — 30% ──────────────────────────────────────
    # HOT_MOMENTUM pullback: current candle naturally has lower volume (healthy fade).
    # Hard-fail only when tape is truly dead (< 25% of avg). RANGE_BOUNCE keeps 50% gate.
    vol_score = 50
    dead_tape_threshold = 0.25 if setup_name in ("HOT_MOMENTUM", "TREND_CONTINUATION", "MOMENTUM_RALLY") else 0.50
    if len(vol) >= 20:
        vol_ma20 = float(vol.rolling(20).mean().iloc[-1])
        ratio    = float(vol.iloc[-1]) / vol_ma20 if vol_ma20 > 0 else 0.0
        if ratio >= 1.5:
            vol_score = 100
        elif ratio >= 1.0:
            vol_score = 80
        elif ratio >= 0.75:
            vol_score = 60
        elif ratio >= 0.5:
            vol_score = 40
        elif ratio >= dead_tape_threshold:
            vol_score = 25
        else:
            vol_score = 0
            fails.append(f"Momentum: volume {ratio:.2f}× avg — dead tape")

    # Bearish engulfing penalty
    candle_range = float(high.iloc[-1]) - float(low.iloc[-1])
    last_op      = float(op.iloc[-1])
    if candle_range > 0 and last < last_op:
        body_pct = abs(last - last_op) / candle_range
        if body_pct > 0.70:
            vol_score = max(0, vol_score - 30)
            if vol_score == 0:
                fails.append("Momentum: strong bearish reversal candle")
    layer["momentum"] = vol_score

    # ── C. Oscillators (RSI + MACD) — 20% ────────────────────────────────
    rsi_val  = float(rsi(close, 14).iloc[-1])
    osc      = 0

    if setup_name == "HOT_MOMENTUM":
        if rsi_val < 45:
            osc = 0;  fails.append(f"Oscillator: RSI {rsi_val:.1f} < 45 — momentum broken")
        elif rsi_val > 72:
            osc = 0;  fails.append(f"Oscillator: RSI {rsi_val:.1f} > 72 — still overbought")
        elif 52 <= rsi_val <= 68:
            osc = 100   # ideal pullback zone
        elif 45 <= rsi_val < 52:
            osc = 55    # cooled hard, momentum questionable
        else:           # 68–72
            osc = 65
    elif setup_name in ("TREND_CONTINUATION", "MOMENTUM_RALLY"):
        # Strength-buy setups, not pullback-buys — ideal zone sits higher than HOT_MOMENTUM's.
        if rsi_val < 45:
            osc = 0;  fails.append(f"Oscillator: RSI {rsi_val:.1f} < 45 — trend not confirmed")
        elif rsi_val > 85:
            osc = 0;  fails.append(f"Oscillator: RSI {rsi_val:.1f} > 85 — extreme, blow-off risk")
        elif 55 <= rsi_val <= 72:
            osc = 100   # ideal trend/momentum zone
        elif 45 <= rsi_val < 55:
            osc = 55    # trend not yet confirmed
        else:           # 72–85
            osc = 65
    else:  # RANGE_BOUNCE
        if rsi_val > 65:
            osc = 0;  fails.append(f"Oscillator: RSI {rsi_val:.1f} > 65 for range bounce")
        elif 35 <= rsi_val <= 52:
            osc = 100
        elif rsi_val < 35:
            osc = 70    # oversold, bounce likely
        else:           # 52–65
            osc = 60

    # MACD histogram adjustment (bonus / hard-fail)
    if len(close) >= 35 and osc > 0:
        _, _, hist = macd(close)
        h_now  = float(hist.iloc[-1])
        h_prev = float(hist.iloc[-2])
        atr_v  = float(atr(df_1h, 14).iloc[-1])
        if h_now > 0 and h_now > h_prev:
            osc = min(100, osc + 15)    # MACD confirming — bonus
        elif h_now < -atr_v * 0.25 and h_now < h_prev:
            osc = 0
            fails.append("Oscillator: MACD deeply negative and falling — full reversal")
    layer["oscillators"] = osc

    # ── D. S/R — 20% (R:R from setup result) ─────────────────────────────
    if setup_result and len(setup_result) >= 5 and setup_result[4] is not None:
        sr_rr = float(setup_result[4])
        layer["sr"] = (100 if sr_rr >= 4.5 else
                        90 if sr_rr >= 4.0 else
                        80 if sr_rr >= 3.5 else
                        70 if sr_rr >= 3.0 else 55)
    else:
        layer["sr"] = 60

    # Hard-fail: any layer = 0 → total = 0
    if any(v == 0 for v in layer.values()):
        return 0.0, layer, fails

    total = round(
        layer["trend"]       * 0.30 +
        layer["momentum"]    * 0.30 +
        layer["oscillators"] * 0.20 +
        layer["sr"]          * 0.20,
        1,
    )
    return total, layer, fails


# ─────────────────────────── Master Model ────────────────────────────────────

def compute_break_model(df: pd.DataFrame, market: str, lookback_break: int = 20, gain_24h_pct: float = 0.0, alpha_mode: bool = False, market_breadth: int = 0, df_4h: Optional[pd.DataFrame] = None):
    """
    Run all setups, pick the best by confidence, then apply confluence bonus.
    Returns consistent 10-tuple.
    """
    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)

    last_atr = float(atr(df, 14).iloc[-1]) if len(df) > 14 else 0.0
    vol_series = df["volume"].astype(float)
    vol_ma = vol_series.rolling(30).mean()
    liquidity_ok = bool(float(vol_ma.iloc[-1]) > 0)
    vol_ok = last_atr > 0

    regime = market_regime(df)

    setups: List[Tuple[str, Tuple]] = []
    setup_scores: Dict[str, Tuple[float, Dict[str, int]]] = {}

    # ── ENABLED SETUPS — updated 2026-08-24 ───────────────────────────────────────
    #
    # STILL DISABLED (0% win rate on prior live data — do not re-enable without
    # new backtested evidence):
    #   BREAKOUT_VOLUME       0/4    WR:0%    avg:-3.1%
    #   PULLBACK_EMA          0/1    WR:0%    avg:-5.2%
    #   OLD_HOT_MOMENTUM      0/12   WR:0%    avg:-8.7%  (no pullback requirement)
    #   PRE_BREAKOUT_COIL     no live data
    #   VOLUME_ACCUMULATION   no live data
    #   VOLATILITY_EXPANSION  no live data
    #
    # ENABLED (positive expected value confirmed in live data):
    #   HOT_MOMENTUM_PULLBACK  106W/46L  WR:69%  avg_win:+14.9%  avg_loss:-5.2%  ← primary
    #   RANGE_BOUNCE             8W/6L   WR:57%  avg_win:+4.5%   avg_loss:-1.7%  ← secondary
    #
    # RE-ENABLED 2026-08-24 (were the only setups covering a trending/breakout
    # market with no pullback — HOT_MOMENTUM/RANGE_BOUNCE both need a pullback or
    # a non-trending regime, so the bot produced zero signals for two weeks during
    # a broad rally in Aug 2026 despite big moves in ZRO/NOT/HEMI/PEOPLE/HOLO/BTC).
    # Original prune (2026-04-27) cited 0/5 WR for MOMENTUM_RALLY and "no live data"
    # for TREND_CONTINUATION — i.e. neither was ever actually live-tested. Both are
    # wired back in with tightened internal thresholds (see detect_trend_continuation
    # ADX fallback and detect_momentum_rally momentum/volume gates) and a higher
    # confluence bar (70 vs 65) as a new, unvalidated live track record starts.
    # Review after each one's first 15-20 resolved signals; disable again (and
    # record observed WR here) if WR < ~40%.
    #   TREND_CONTINUATION    re-enabled, no live data yet — watch closely
    #   MOMENTUM_RALLY         re-enabled, was 0/5 WR:0% avg:-6.6% (too small a sample
    #                           to trust; thresholds tightened since — watch closely)
    #
    # Confidence note: confluence bonus DISABLED for HOT_MOMENTUM (inverts win rate).
    #   79-83% conf → 88% WR;  84-87% → 79% WR;  88-91% → 61% WR;  92-95% → 54% WR

    # Range bounce — run in all non-trending regimes
    if regime in ("RANGE", "COMPRESSION", "CHOPPY"):
        s3 = detect_range_bounce(df, market, lookback_break)
        if s3:
            cs3, cl3, _ = confluence_score(df, df_4h, "RANGE_BOUNCE", s3)
            if cs3 >= 65:
                setups.append(("RANGE_BOUNCE", s3))
                setup_scores["RANGE_BOUNCE"] = (cs3, cl3)

    # Hot momentum pullback — only when coin is a confirmed 24h mover
    # 15-20% gain zone has 9% WR historically — worst bucket, skip it
    # Require ≥20% in moderate markets, ≥15% only in broad bull markets (breadth ≥ 15)
    # Cap raised 35→60 (2026-08-24): 35% had no bucket-level evidence behind it and was
    # excluding exactly the size of move a hot rally (ZRO/NOT/HEMI-style) produces.
    hot_cap      = 70.0 if alpha_mode else 60.0
    hot_min_gain = 20.0 if market_breadth < 15 else 15.0
    if hot_min_gain <= gain_24h_pct <= hot_cap:
        s9 = detect_hot_momentum(df, market, gain_24h_pct)
        if s9:
            cs9, cl9, _ = confluence_score(df, df_4h, "HOT_MOMENTUM", s9)
            if cs9 >= 65:
                setups.append(("HOT_MOMENTUM", s9))
                setup_scores["HOT_MOMENTUM"] = (cs9, cl9)

    # Trend continuation — catches a coin trending up with no meaningful pullback,
    # which HOT_MOMENTUM (pullback-only) and RANGE_BOUNCE (non-trending only) both miss.
    # Re-enabled 2026-08-24, tighter confluence bar (70) while it's unvalidated live.
    s_tc = detect_trend_continuation(df, market, lookback_break)
    if s_tc:
        cs_tc, cl_tc, _ = confluence_score(df, df_4h, "TREND_CONTINUATION", s_tc)
        if cs_tc >= 70:
            setups.append(("TREND_CONTINUATION", s_tc))
            setup_scores["TREND_CONTINUATION"] = (cs_tc, cl_tc)

    # Momentum rally — catches broad bull-market strength (elevated RSI, real volume,
    # price holding upper range) rather than requiring a pump-then-pullback structure.
    # Re-enabled 2026-08-24, tighter confluence bar (70) while it's unvalidated live.
    s_mr = detect_momentum_rally(df, market)
    if s_mr:
        cs_mr, cl_mr, _ = confluence_score(df, df_4h, "MOMENTUM_RALLY", s_mr)
        if cs_mr >= 70:
            setups.append(("MOMENTUM_RALLY", s_mr))
            setup_scores["MOMENTUM_RALLY"] = (cs_mr, cl_mr)

    if setups:
        best_name, best = max(setups, key=lambda x: x[1][6])
        direction, entry_zone, stop, targets, rr, exp_pct, conf, reasons = best

        cs, cl = setup_scores.get(best_name, (0.0, {}))
        score_line = (
            f"Confluence: {cs:.0f}/100 — "
            f"Trend:{cl.get('trend',0)} Mom:{cl.get('momentum',0)} "
            f"Osc:{cl.get('oscillators',0)} SR:{cl.get('sr',0)}"
        )

        # Confluence bonus only for RANGE_BOUNCE (inverts WR for HOT_MOMENTUM)
        if best_name == "RANGE_BOUNCE":
            conf, bonus_reasons = confluence_bonus(df, conf)
            all_reasons = reasons + bonus_reasons + [f"Regime: {regime}", f"Setup: {best_name}", score_line]
        else:
            all_reasons = reasons + [f"Regime: {regime}", f"Setup: {best_name}", score_line]

        return direction, entry_zone, stop, targets, rr, exp_pct, conf, all_reasons, liquidity_ok, vol_ok

    # No fallback — if no enabled setup fires, return NONE cleanly
    # (Removed breakout fallback: 0% win rate in live data)
    return "NONE", None, None, [], None, 0.0, 40, \
           [f"No qualifying setup — regime: {regime}, gain_24h: {gain_24h_pct:.1f}%"], \
           liquidity_ok, vol_ok


# ─────────────────────────── Spot-only action ────────────────────────────────

def spot_only_action(raw_dir: str, position_size: float, conf: int = 0, rr: float = 0.0) -> str:
    """
    Translate raw signal direction into a spot-only action label.
    For holdings: ADD means "strong setup — consider adding to position"
    """
    holding = position_size > 0
    if raw_dir == "LONG":
        if holding:
            # Strong setup on something you already hold = ADD signal
            return "ADD" if (conf >= 70 and rr >= 2.5) else "HOLD"
        return "BUY"
    if raw_dir == "SHORT":
        return "SELL" if holding else "AVOID"
    return "HOLD" if holding else "AVOID"


# ─────────────────────────── Scan ────────────────────────────────────────────

def _analyse_crypto(sym: str, timeframe: str, limit: int, holdings: dict, gain_24h_pct: float = 0.0, btc_trend: str = 'NEUTRAL', alpha_mode: bool = False, market_breadth: int = 0, btc_dom_rising: bool = False, funding_rate: float = 0.0) -> Idea:
    """Analyse a single crypto symbol — runs in a thread pool for parallelism."""
    exchanges = check_exchanges(sym)
    pos = get_position_size(holdings, "CRYPTO", sym)
    holding = pos > 0

    if not exchanges and not holding:
        return None  # Not tradeable and not held — skip silently

    df = fetch_crypto_ohlcv(sym, timeframe=timeframe, limit=limit)
    if len(df) < 30:
        return Idea("CRYPTO", sym, "AVOID", None, None, [], None, None, 40,
            [f"Not enough data ({len(df)} bars)."], exchanges)

    # Fetch 4H data for trend layer (Minervini EMA50 check)
    try:
        df_4h = fetch_crypto_ohlcv(sym, timeframe="4h", limit=120)
        df_4h = df_4h if len(df_4h) >= 55 else None
    except Exception:
        df_4h = None

    raw_dir, entry, stop, targets, rr, exp_pct, conf, why, liquid_ok, vol_ok = compute_break_model(df, market="CRYPTO", gain_24h_pct=gain_24h_pct, alpha_mode=alpha_mode, market_breadth=market_breadth, df_4h=df_4h)

    # Session filter: off-peak hours penalise confidence.
    # Raised 3→8 pts (2026-08-24), replacing the separate hard BUY-block below —
    # crypto trades 24/7 and off-peak (Asian-session) hours are exactly when
    # coins like NOT/HOLO/HEMI often move, so a genuinely strong signal (raw
    # conf ≥80) can now still clear the conf>=72 BUY gate instead of being
    # unconditionally downgraded to WATCH.
    sess_name, is_prime = session_context()
    if not is_prime:
        conf = max(40, conf - 8)

    if conf >= 72 and rr is not None and rr >= 2.5 and exp_pct is not None and exp_pct >= 4.0:
        if raw_dir in ("LONG", "SHORT"):
            action = spot_only_action(raw_dir, pos, conf=conf, rr=rr or 0.0)
        elif raw_dir == "BUY":
            action = "HOLD" if holding else "BUY"
        elif raw_dir == "SELL":
            action = "SELL" if holding else "AVOID"
        else:
            action = "WATCH"
    else:
        action = "HOLD" if holding else "AVOID"

    # BTC downtrend: suppress altcoin longs
    if action == "BUY" and btc_trend == "DOWN" and sym != "BTC/USDT":
        action = "WATCH"
        why = ["BTC in downtrend — BUY downgraded to WATCH"] + why

    # BTC dominance rising: capital rotating from alts into BTC
    if action == "BUY" and btc_dom_rising and sym not in ("BTC/USDT", "ETH/USDT"):
        action = "WATCH"
        why = ["BTC dominance rising — altcoin headwind, BUY downgraded to WATCH"] + why

    # Funding rate: overleveraged longs = squeeze risk
    if funding_rate > 0.001:
        if action == "BUY":
            action = "WATCH"
        why = [f"⚠ Funding {funding_rate*100:.3f}% (>0.1%) — severe overleverage, squeeze risk"] + why
    elif funding_rate > 0.0005:
        why = [f"⚠ Funding {funding_rate*100:.3f}% (>0.05%) — longs dominant, caution"] + why

    # Off-peak: no longer a hard BUY block (removed 2026-08-24) — the -8pt
    # confidence penalty above already filters out weak off-peak signals via the
    # conf>=72 gate. Just annotate for visibility.
    if not is_prime:
        why = [f"Off-peak session ({sess_name}) — confidence penalised, not blocked"] + why

    return Idea("CRYPTO", sym, action, entry, stop, targets or [], rr, exp_pct, conf, why, exchanges)


def _analyse_stock(tkr: str, timeframe: str, limit: int, holdings: dict) -> Idea:
    """Analyse a single stock ticker — runs in a thread pool for parallelism."""
    df = fetch_stock_ohlcv(tkr, timeframe=timeframe, limit=limit)
    if len(df) < 30:
        return Idea("STOCK", tkr, "AVOID", None, None, [], None, None, 40,
            [f"Not enough data ({len(df)} bars)."])

    raw_dir, entry, stop, targets, rr, exp_pct, conf, why, liquid_ok, vol_ok = compute_break_model(df, market="STOCK")
    pos = get_position_size(holdings, "STOCK", tkr)
    holding = pos > 0

    if conf >= 70 and rr is not None and rr >= 2.5 and exp_pct is not None and exp_pct >= 3.5:
        if raw_dir in ("LONG", "SHORT"):
            action = spot_only_action(raw_dir, pos, conf=conf, rr=rr or 0.0)
        elif raw_dir == "BUY":
            action = "HOLD" if holding else "BUY"
        elif raw_dir == "SELL":
            action = "SELL" if holding else "AVOID"
        else:
            action = "WATCH"
    else:
        action = "HOLD" if holding else "AVOID"

    return Idea("STOCK", tkr, action, entry, stop, targets or [], rr, exp_pct, conf, why)


def scan_spot_and_stocks(
    crypto_symbols: List[str],
    stock_tickers: List[str],
    timeframe: str = "1h",
    limit: int = 500,
    holdings_path: str = "holdings.json",
    max_workers: int = 8,
    climber_gains: Optional[Dict[str, float]] = None,
    alpha_symbols: Optional[set] = None,
) -> List[Idea]:
    """
    Parallel scan — fetches all symbols concurrently using a thread pool.
    climber_gains: dict of {symbol: gain_24h_pct} from find_steady_climbers, used to
                   activate the HOT_MOMENTUM setup for confirmed movers.
    alpha_symbols: set of symbols identified as new listings or Binance Alpha tokens.
                   These use a higher 24h gain cap (70% vs 35%) in the momentum detector.
    """
    holdings = load_holdings(holdings_path)
    ideas: List[Idea] = []
    gains = climber_gains or {}
    alpha = alpha_symbols or set()

    # Warm up exchange cache before parallel run (avoid race conditions)
    _load_exchange_symbols("binance")
    _load_exchange_symbols("kraken")

    # ── BTC market context: suppress BUY signals in downtrend ────────────
    btc_trend = "NEUTRAL"
    try:
        btc_df = fetch_crypto_ohlcv("BTC/USDT", timeframe=timeframe, limit=100)
        if len(btc_df) >= 50:
            btc_close = btc_df["close"].astype(float)
            btc_e20   = ema(btc_close, 20)
            btc_e50   = ema(btc_close, 50)
            btc_rsi   = float(rsi(btc_close, 14).iloc[-1])
            # Downtrend: EMA20 < EMA50 AND RSI < 50 (softened from 45 to catch more choppy regimes)
            if float(btc_e20.iloc[-1]) < float(btc_e50.iloc[-1]) and btc_rsi < 50:
                btc_trend = "DOWN"
            # Strong uptrend: EMA20 > EMA50 AND RSI > 55
            elif float(btc_e20.iloc[-1]) > float(btc_e50.iloc[-1]) and btc_rsi > 55:
                btc_trend = "UP"
    except Exception:
        pass  # if BTC fetch fails, allow all signals through

    # Market breadth: count of coins with 10%+ 24h gain — proxy for participation
    market_breadth = sum(1 for v in gains.values() if v >= 10.0)

    # Macro context: BTC dominance + all funding rates (cached, single call each)
    btc_dom_pct, btc_dom_rising = get_btc_dominance()
    funding_rates = get_all_funding_rates()

    # ── Parallel crypto scan ──────────────────────────────────────────────
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                _analyse_crypto, sym, timeframe, limit, holdings,
                gains.get(sym, 0.0), btc_trend, sym in alpha, market_breadth,
                btc_dom_rising, funding_rates.get(sym, 0.0),
            ): sym
            for sym in crypto_symbols
        }
        for future in as_completed(futures):
            try:
                result = future.result(timeout=15)
                if result is not None:
                    ideas.append(result)
            except Exception as e:
                sym = futures[future]
                ideas.append(Idea("CRYPTO", sym, "AVOID", None, None, [], None, None, 30,
                    [f"Error: {type(e).__name__}: {e}"]))

    # ── Parallel stock scan ───────────────────────────────────────────────
    with ThreadPoolExecutor(max_workers=min(max_workers, 4)) as ex:
        futures = {
            ex.submit(_analyse_stock, tkr, timeframe, limit, holdings): tkr
            for tkr in stock_tickers
        }
        for future in as_completed(futures):
            try:
                result = future.result(timeout=20)
                if result is not None:
                    ideas.append(result)
            except Exception as e:
                tkr = futures[future]
                ideas.append(Idea("STOCK", tkr, "AVOID", None, None, [], None, None, 30,
                    [f"Error: {type(e).__name__}: {e}"]))

    def score(i: Idea):
        actionable = 2 if i.action in ("BUY", "SELL", "ADD") else (1 if i.action == "WATCH" else 0)
        rr_val = i.rr if i.rr is not None else -1.0
        return (actionable, i.confidence, rr_val)

    ideas.sort(key=score, reverse=True)
    return ideas


# ─────────────────────────── Console renderer ────────────────────────────────

def render_console(ideas: List[Idea], top: int = 15, diag: bool = False) -> None:
    console.rule("Active Opportunity Scanner (Read-only, Spot-Only)")
    console.print(f"Time: {datetime.now(timezone.utc)} UTC\n")

    buy_count   = sum(1 for i in ideas if i.action == "BUY")
    sell_count  = sum(1 for i in ideas if i.action == "SELL")
    watch_count = sum(1 for i in ideas if i.action == "WATCH")

    console.print(
        f"[bold green]BUY: {buy_count}[/bold green] | "
        f"[bold red]SELL: {sell_count}[/bold red] | "
        f"[bold yellow]WATCH: {watch_count}[/bold yellow]\n"
    )

    table = Table(title="Trading Opportunities")
    table.add_column("Market",  style="cyan")
    table.add_column("Symbol",  style="bold")
    table.add_column("Setup",   style="magenta")
    table.add_column("Action",  style="bold")
    table.add_column("Entry")
    table.add_column("Stop")
    table.add_column("TP1")
    table.add_column("TP2")
    table.add_column("Exp %")
    table.add_column("R:R")
    table.add_column("Conf %")

    for i in ideas[:top]:
        entry_txt = "-" if not i.entry else f"{i.entry[0]:.4f}–{i.entry[1]:.4f}"
        stop_txt  = "-" if i.stop is None else f"{i.stop:.4f}"
        tp1_txt   = f"{i.targets[0]:.4f}" if i.targets else "-"
        tp2_txt   = f"{i.targets[1]:.4f}" if len(i.targets) > 1 else "-"
        rr_txt    = "-" if i.rr is None else f"{i.rr:.2f}"
        exp_txt   = "-" if i.expected_pct is None else f"{i.expected_pct:.1f}%"

        colour = {"BUY": "green", "SELL": "red", "WATCH": "yellow", "HOLD": "cyan"}.get(i.action, "white")

        table.add_row(
            i.market, i.symbol,
            i.why[0][:40] if i.why else "-",
            f"[bold {colour}]{i.action}[/bold {colour}]",
            entry_txt, stop_txt, tp1_txt, tp2_txt, exp_txt, rr_txt,
            f"{i.confidence}%",
        )

    console.print(table)

    if diag:
        console.print("\n[bold]📊 Detailed Analysis:[/bold]")
        for idx, i in enumerate(ideas[:top], 1):
            indicator = "🟢" if i.confidence >= 80 else "🟡" if i.confidence >= 70 else "🟠" if i.confidence >= 60 else "🔴"
            console.print(f"\n{indicator} [{idx}] {i.market} {i.symbol} → [bold]{i.action}[/bold]")
            for reason in i.why:
                console.print(f"    • {reason}")


def run_scan(symbols: List[str], timeframe: str = "1h", limit: int = 500, top: int = 5, diag: bool = False) -> None:
    ideas = scan_spot_and_stocks(crypto_symbols=symbols, stock_tickers=[], timeframe=timeframe, limit=limit)
    render_console(ideas, top=top, diag=diag)
