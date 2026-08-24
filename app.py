import os
import sys
import time
import json
import uuid
import streamlit as st
import pandas as pd
from datetime import datetime, timezone
from typing import List

sys.path.insert(0, os.path.abspath("src"))

from bot.scanner import (
    scan_spot_and_stocks,
    top_binance_spot_symbols,
    find_steady_climbers,
    fetch_crypto_ohlcv,
    update_listings_cache,
    get_new_listings,
    get_btc_dominance,
    session_context,
    DEFAULT_SYMBOLS,
    DEFAULT_STOCKS,
)

st.set_page_config(page_title="Portfolio Advisor", layout="wide")
st.title("Portfolio Advisor (Local, Read-only)")
st.caption("Spot-only: BUY / SELL / HOLD / AVOID. No futures, no shorting, no order execution.")
st.caption("Opportunities filtered to Binance / Kraken only.")

ARCHIVE_PATH = "trade_archive.json"
DEFAULT_CRYPTO = DEFAULT_SYMBOLS


# ─────────────────────────── Archive helpers ─────────────────────────────────

def load_archive() -> list:
    try:
        with open(ARCHIVE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_archive(records: list):
    with open(ARCHIVE_PATH, "w", encoding="utf-8") as f:
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


# ─────────────────────────── Holdings helper ─────────────────────────────────

def load_holdings(path: str = "holdings.json"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"crypto": {}, "stocks": {}}


# ─────────────────────────── UI controls ─────────────────────────────────────

crypto_symbols_text = st.text_area("Crypto symbols (comma-separated)", value=",".join(DEFAULT_CRYPTO), height=80)
stocks_text = st.text_area("Stocks/ETFs tickers (comma-separated)", value=",".join(DEFAULT_STOCKS), height=60)

col0, col1, col2, col3 = st.columns(4)
with col0:
    use_top = st.checkbox("Auto top Binance USDT markets", value=True)
with col1:
    use_stock_basket = st.checkbox("Use US large-cap basket", value=True)
with col2:
    crypto_tf = st.selectbox("Crypto timeframe", ["15m", "1h", "4h"], index=1)
with col3:
    stock_tf = st.selectbox("Stocks/ETF timeframe", ["1d", "4h", "1h"], index=0)

refresh          = st.slider("Refresh (seconds)", 30, 3600, 300)
climber_min_gain = st.slider("Rising coins — min 24h gain %", 3, 30, 5)
climber_min_vol  = st.slider("Rising coins — min 24h volume ($k USDT)", 100, 5000, 200) * 1000
top_market_count = st.slider("Top crypto markets to scan", 20, 400, 80, 10)
scan_all_crypto  = st.checkbox("Scan ALL Binance USDT spot markets (slow)", value=False)
limit            = st.slider("Candles to fetch", 100, 500, 250, 50)
top              = st.slider("Rows to show", 5, 50, 25)
show_diag        = st.checkbox("Show diagnostics (why)", value=False)
max_crypto       = st.slider("Max crypto symbols per scan", 10, 600, 40, 10)
max_stocks       = st.slider("Max stock/ETF symbols per scan", 5, 150, 5, 5)

st.divider()
st.subheader("🆕 Binance Alpha / New Listings")
st.caption(
    "New listings detected automatically via symbol diff (first time a ticker appears on Binance spot). "
    "Alpha tokens scanned with a 70% 24h gain cap (vs 35% for regular coins) to catch post-listing surges."
)
col_a, col_b = st.columns(2)
with col_a:
    new_listing_days = st.slider("New listing lookback (days)", 7, 90, 30)
with col_b:
    alpha_manual_text = st.text_input(
        "Manual Alpha watchlist (comma-separated, e.g. USELESS/USDT,BANANA/USDT)",
        value="",
        placeholder="TOKEN/USDT, TOKEN2/USDT",
    )

alpha_manual_symbols = {s.strip().upper() for s in alpha_manual_text.split(",") if s.strip()}

crypto_symbols  = [s.strip() for s in crypto_symbols_text.split(",") if s.strip()]
stock_tickers   = [s.strip() for s in stocks_text.split(",") if s.strip()]
holdings        = load_holdings()
holdings_crypto = list(holdings.get("crypto", {}).keys())
holdings_stocks = list(holdings.get("stocks", {}).keys())

placeholder = st.empty()

if "loop_idx" not in st.session_state:
    st.session_state.loop_idx = 0

# ─────────────────────────── Main scan loop ───────────────────────────────────

while True:
    st.session_state.loop_idx += 1
    _k = st.session_state.loop_idx  # unique key suffix for this iteration
    symbols = crypto_symbols
    all_binance_symbols: List[str] = []
    if use_top:
        try:
            all_binance_symbols = top_binance_spot_symbols(limit=None)
            symbols = all_binance_symbols[:top_market_count] if not scan_all_crypto else all_binance_symbols
        except Exception as e:
            st.warning(f"Could not load top markets: {e}")

    # Update listings cache with the full symbol universe and detect new ones
    if all_binance_symbols:
        update_listings_cache(all_binance_symbols)
        new_listing_hits = get_new_listings(all_binance_symbols, days=new_listing_days)
    else:
        new_listing_hits = []
    new_listing_symbols = {n["symbol"] for n in new_listing_hits}

    # Merge: manual alpha + auto-detected new listings
    alpha_symbols_all = alpha_manual_symbols | new_listing_symbols

    base_crypto = [s for s in symbols if s not in holdings_crypto]
    # Auto-inject alpha symbols into the scan even if they're outside the top-N
    alpha_inject = [s for s in alpha_symbols_all if s not in symbols and s not in holdings_crypto]
    symbols = (holdings_crypto + base_crypto[:max_crypto] + alpha_inject)[:max_crypto + len(alpha_inject)]

    stocks = DEFAULT_STOCKS if use_stock_basket else stock_tickers
    stocks = (holdings_stocks + [s for s in stocks if s not in holdings_stocks][:max_stocks])[:max_stocks]

    try:
        climbers = find_steady_climbers(min_gain_pct=climber_min_gain, min_vol_usdt=climber_min_vol)
    except Exception:
        climbers = []

    climber_gains = {c["symbol"]: c["gain_pct"] for c in climbers}
    market_breadth = sum(1 for v in climber_gains.values() if v >= 10.0)
    for cs in [c["symbol"] for c in climbers[:20]]:
        if cs not in symbols:
            symbols.append(cs)

    try:
        crypto_ideas = scan_spot_and_stocks(
            crypto_symbols=symbols, stock_tickers=[],
            timeframe=crypto_tf, limit=limit,
            holdings_path="holdings.json", climber_gains=climber_gains,
            alpha_symbols=alpha_symbols_all,
        )
    except Exception as e:
        st.error(f"Crypto scan failed: {e}")
        crypto_ideas = []

    try:
        stock_ideas = scan_spot_and_stocks(
            crypto_symbols=[], stock_tickers=stocks,
            timeframe=stock_tf, limit=limit, holdings_path="holdings.json",
        )
    except Exception as e:
        st.error(f"Stock scan failed: {e}")
        stock_ideas = []

    ideas = crypto_ideas + stock_ideas
    ideas.sort(key=lambda i: (1 if i.action in ("BUY","SELL") else 0, i.confidence, i.rr or -1), reverse=True)

    holdings_ideas = [i for i in ideas if
        (i.market == "CRYPTO" and i.symbol in holdings_crypto) or
        (i.market == "STOCK"  and i.symbol in holdings_stocks)]

    opp_ideas = [i for i in ideas if
        i.action in ("BUY", "WATCH", "ADD") and
        (getattr(i, "expected_pct", 0.0) or 0.0) >= 3.0 and
        len(getattr(i, "exchanges", []) or []) > 0]

    # ── Archive: log new BUY signals, auto-resolve pending ───────────────
    archive = load_archive()
    # Cap BUY signals to top 3 per scan (by confidence then R:R) to avoid correlated batch losses
    _buy_candidates = sorted(
        [i for i in opp_ideas if i.action == "BUY"],
        key=lambda x: (x.confidence, x.rr or 0), reverse=True
    )[:3]
    _other_signals = [i for i in opp_ideas if i.action != "BUY"]
    new_count = sum(1 for i in _buy_candidates + _other_signals if i.action in ("BUY","ADD") and archive_signal(i, archive))
    resolve_pending(archive)
    save_archive(archive)
    stats = reliability_stats(archive)

    # Hide opportunities that already resolved (hit TP or stopped) in the last 36h
    _now_ts = time.time()
    recently_resolved = {
        r["symbol"] for r in archive
        if r["outcome"] in ("HIT_TP1", "HIT_TP2", "STOPPED_OUT")
        and r.get("resolved_at")
        and (_now_ts - datetime.fromisoformat(r["resolved_at"]).timestamp()) < 36 * 3600
    }
    opp_ideas = [i for i in opp_ideas if i.symbol not in recently_resolved]

    def to_rows(items):
        rows = []
        for i in items:
            exp  = getattr(i, "expected_pct", None)
            exch = ", ".join(getattr(i, "exchanges", []) or [])
            rows.append({
                "Market": i.market, "Symbol": i.symbol, "Exchange": exch or "—",
                "Action": i.action,
                "Entry Low":  None if not i.entry else round(i.entry[0], 4),
                "Entry High": None if not i.entry else round(i.entry[1], 4),
                "Stop":       None if i.stop is None else round(i.stop, 4),
                "Targets":    "" if not i.targets else ", ".join(f"{t:.4f}" for t in i.targets),
                "R:R":        None if i.rr is None else round(i.rr, 2),
                "Expected %": None if exp is None else round(exp, 1),
                "Confidence": f"{i.confidence}%",
            })
        return pd.DataFrame(rows)

    with placeholder.container():
        dom_pct, dom_rising = get_btc_dominance()
        sess_name, is_prime  = session_context()
        dom_tag   = f"↑🔴 {dom_pct:.1f}%" if dom_rising else f"→ {dom_pct:.1f}%"
        sess_tag  = f"🟢 {sess_name}" if is_prime else f"🟡 {sess_name} (−5 conf)"
        st.caption(
            f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')} | TF: {crypto_tf} | "
            f"New signals: {new_count} | BTC Dom: {dom_tag} | Session: {sess_tag}"
        )

        # ── Reliability metrics ───────────────────────────────────────────
        breadth_label = "🟢 Active" if market_breadth >= 20 else "🟡 Choppy"
        dom_warn = " | ⚠ BTC dominance rising — altcoin longs suppressed" if dom_rising else ""
        st.caption(
            f"Market breadth: **{market_breadth}** coins with 10%+ gain — {breadth_label}"
            f" | Confluence min: **75/100** required{dom_warn}"
        )
        if stats["total"] > 0:
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Signals logged", stats["total"])
            c2.metric("Win rate",        f"{stats['win_rate']}%")
            c3.metric("Avg result",      f"{stats['avg_gain']}%")
            c4.metric("TP1 capture",     f"{stats['avg_capture']}%",
                      help="Avg % of expected move captured at TP1 (42% = typical)")
            c5.metric("Pending",         stats["pending"])
            c6.metric("W / L",           f"{stats['wins']} / {stats['losses']}")

        # ── Holdings ─────────────────────────────────────────────────────
        st.subheader("Holdings — guidance")
        st.dataframe(to_rows(holdings_ideas[:top]), use_container_width=True, hide_index=True)

        actionable = [i for i in holdings_ideas if
            i.action in ("ADD","BUY") or
            (i.action == "HOLD" and i.entry is not None and i.confidence >= 65)]
        if actionable:
            st.warning(f"⚡ {len(actionable)} holding(s) with active signal:")
            for sig in actionable:
                exch  = ", ".join(getattr(sig, "exchanges", []) or [])
                badge = "🟢 ADD" if sig.action == "ADD" else "🟢 BUY" if sig.action == "BUY" else "🔵 HOLD (active)"
                entry_str = f"`{sig.entry[0]:.4f}–{sig.entry[1]:.4f}`" if sig.entry else "—"
                st.markdown(f"{badge} **{sig.symbol}** on {exch or '—'} | Entry: {entry_str} | Conf: `{sig.confidence}%`")
                if sig.why:
                    st.caption(f"↳ {sig.why[0]}")

        # ── Opportunities ─────────────────────────────────────────────────
        st.subheader("Opportunities — potential entries (Binance / Kraken only)")
        # Tag alpha/new-listing signals with a badge
        alpha_badge_syms = alpha_symbols_all
        df_opps_raw = opp_ideas[:top]
        if df_opps_raw:
            rows_opps = []
            for i in df_opps_raw:
                exp  = getattr(i, "expected_pct", None)
                exch = ", ".join(getattr(i, "exchanges", []) or [])
                tag  = "🆕 NEW" if i.symbol in new_listing_symbols else ("⭐ ALPHA" if i.symbol in alpha_manual_symbols else "")
                rows_opps.append({
                    "Tag":        tag,
                    "Market":     i.market, "Symbol": i.symbol, "Exchange": exch or "—",
                    "Action":     i.action,
                    "Entry Low":  None if not i.entry else round(i.entry[0], 4),
                    "Entry High": None if not i.entry else round(i.entry[1], 4),
                    "Stop":       None if i.stop is None else round(i.stop, 4),
                    "Targets":    "" if not i.targets else ", ".join(f"{t:.4f}" for t in i.targets),
                    "R:R":        None if i.rr is None else round(i.rr, 2),
                    "Expected %": None if exp is None else round(exp, 1),
                    "Confidence": f"{i.confidence}%",
                })
            st.dataframe(pd.DataFrame(rows_opps), use_container_width=True, hide_index=True)
        else:
            st.info("No qualifying opportunities right now.")

        # ── New Binance Listings / Alpha ──────────────────────────────────
        st.subheader("🆕 New Binance Listings — auto-detected & Alpha watchlist")
        st.caption(
            f"Symbols first seen on Binance in the last {new_listing_days} days are auto-injected into the scan "
            "with a 70% gain cap (vs 35% for regular coins). Add manual Alpha tokens above."
        )
        alpha_rows = []
        for n in new_listing_hits:
            gain = climber_gains.get(n["symbol"])
            alpha_rows.append({
                "Symbol":       n["symbol"],
                "First seen":   f"{n['days_ago']}d ago",
                "In scan":      "✅" if n["symbol"] in set(symbols) else "⚡ injected",
                "24h Gain %":   f"+{gain:.1f}%" if gain else "—",
                "Signal":       next((f"{i.action} {i.confidence}%" for i in crypto_ideas if i.symbol == n["symbol"] and i.action not in ("AVOID","HOLD","WATCH")), "—"),
            })
        for sym in sorted(alpha_manual_symbols):
            gain = climber_gains.get(sym)
            if not any(r["Symbol"] == sym for r in alpha_rows):
                alpha_rows.append({
                    "Symbol":     sym,
                    "First seen": "manual",
                    "In scan":    "✅" if sym in set(symbols) else "⚡ injected",
                    "24h Gain %": f"+{gain:.1f}%" if gain else "—",
                    "Signal":     next((f"{i.action} {i.confidence}%" for i in crypto_ideas if i.symbol == sym and i.action not in ("AVOID","HOLD","WATCH")), "—"),
                })
        if alpha_rows:
            st.dataframe(pd.DataFrame(alpha_rows), use_container_width=True, hide_index=True)
        else:
            st.info(f"No new listings detected in the last {new_listing_days} days. Cache builds up over time — new symbols are captured on the next scan after they appear on Binance.")

        # ── Rising coins ──────────────────────────────────────────────────
        st.subheader("🚀 Rising Coins — Binance 24h gainers")
        st.caption("Top 20 auto-injected into full scan. Check Opportunities for entry details.")
        if climbers:
            already = set(symbols)
            st.dataframe(pd.DataFrame([{
                "Symbol":       c["symbol"],
                "Status":       "✅ Fully analysed" if c["symbol"] in already else "⚡ Auto-added",
                "24h Gain %":   f"+{c['gain_pct']:.1f}%",
                "Price (USDT)": c["price"],
                "24h Volume":   f"${c['volume_usdt']:,.0f}",
            } for c in climbers]), use_container_width=True, hide_index=True)
        else:
            st.info("No coins above threshold right now.")

        # ── Top signal ────────────────────────────────────────────────────
        best = next((x for x in opp_ideas if x.action in ("BUY","SELL")), None)
        if best:
            exp  = getattr(best, "expected_pct", None)
            exch = ", ".join(getattr(best, "exchanges", []) or [])
            st.subheader(f"Top actionable: {best.symbol} → {best.action}")
            st.write({"Exchange": exch or "—",
                      "Entry":    "-" if not best.entry else f"{best.entry[0]:.4f}–{best.entry[1]:.4f}",
                      "Stop":     "-" if not best.stop else f"{best.stop:.4f}",
                      "Targets":  ", ".join(f"{t:.4f}" for t in best.targets) if best.targets else "-",
                      "R:R":      "-" if not best.rr else f"{best.rr:.2f}",
                      "Expected": "-" if not exp else f"{exp:.1f}%",
                      "Conf":     f"{best.confidence}%"})
            if show_diag:
                for w in best.why:
                    st.write(f"- {w}")
        else:
            st.subheader("No BUY/SELL right now")
            st.write("Only HOLD/AVOID signals at the moment.")

        # ── Signal Archive ────────────────────────────────────────────────
        st.divider()
        st.subheader("📋 Signal Archive — outcome tracker")
        st.caption("Every BUY signal is auto-logged. Outcomes resolved by price check or mark manually.")

        archive_fresh = load_archive()

        if not archive_fresh:
            st.info("No signals logged yet — BUY signals will appear here automatically.")
        else:
            # Manual resolve
            pending = [r for r in archive_fresh if r["outcome"] == "PENDING"]
            if pending:
                with st.expander(f"✏️ Manually resolve {len(pending)} pending signal(s)"):
                    sel_id  = st.selectbox("Signal ID", [r["id"] for r in pending], key=f"sel_id_{_k}")
                    sel_rec = next((r for r in pending if r["id"] == sel_id), None)
                    if sel_rec:
                        st.write(f"**{sel_rec['symbol']}** | Logged: {sel_rec['logged_at'][:16]} | "
                                 f"Entry: {sel_rec['entry_low']} | TP1: {sel_rec['tp1']} | "
                                 f"TP2: {sel_rec['tp2']} | Stop: {sel_rec['stop']}")
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    clicked = None
                    if mc1.button("✅ Hit TP1"):   clicked = "HIT_TP1"
                    if mc2.button("🎯 Hit TP2"):   clicked = "HIT_TP2"
                    if mc3.button("❌ Stopped"):   clicked = "STOPPED_OUT"
                    if mc4.button("⏰ Expire"):    clicked = "EXPIRED"
                    if clicked and sel_rec:
                        for r in archive_fresh:
                            if r["id"] == sel_id:
                                r["outcome"]     = clicked
                                r["resolved_at"] = datetime.now(timezone.utc).isoformat()
                                entry = r.get("entry_low") or 0
                                if clicked == "HIT_TP1" and r.get("tp1") and entry:
                                    r["outcome_pct"] = round((r["tp1"] - entry) / entry * 100, 1)
                                elif clicked == "HIT_TP2" and r.get("tp2") and entry:
                                    r["outcome_pct"] = round((r["tp2"] - entry) / entry * 100, 1)
                                elif clicked == "STOPPED_OUT" and r.get("stop") and entry:
                                    r["outcome_pct"] = round((r["stop"] - entry) / entry * 100, 1)
                                break
                        save_archive(archive_fresh)
                        st.success(f"Marked {sel_id} as {clicked}")
                        st.rerun()

            # Notes
            with st.expander("📝 Add note to a signal"):
                note_id   = st.selectbox("Signal", [r["id"] for r in archive_fresh], key=f"note_id_{_k}")
                note_text = st.text_input("Note text", key=f"note_txt_{_k}")
                if st.button("Save note"):
                    for r in archive_fresh:
                        if r["id"] == note_id:
                            r["notes"] = note_text
                            break
                    save_archive(archive_fresh)
                    st.success("Saved.")
                    st.rerun()

            # Filter + table
            filter_out = st.multiselect(
                "Filter by outcome",
                ["PENDING","HIT_TP1","HIT_TP2","STOPPED_OUT","EXPIRED","MANUAL_WIN","MANUAL_LOSS"],
                default=["PENDING","HIT_TP1","HIT_TP2","STOPPED_OUT"],
                key=f"archive_filter_{_k}",
            )
            filtered = [r for r in reversed(archive_fresh) if r["outcome"] in filter_out]
            if not filtered:
                st.info("No records match filters.")
            else:
                st.dataframe(pd.DataFrame([{
                    "ID":        r["id"],
                    "Logged":    r["logged_at"][:16].replace("T"," "),
                    "Symbol":    r["symbol"],
                    "Exchange":  r["exchange"],
                    "Action":    r["action"],
                    "Entry":     r["entry_low"],
                    "Stop":      r["stop"],
                    "TP1":       r["tp1"],
                    "TP2":       r["tp2"],
                    "Exp %":     r["expected_pct"],
                    "Conf":      r["confidence"],
                    "Outcome":   outcome_badge(r["outcome"]),
                    "Result %":  r["outcome_pct"],
                    "Resolved":  (r["resolved_at"] or "")[:16].replace("T"," "),
                    "Setup":     r["setup"][:45] if r["setup"] else "",
                    "Notes":     r["notes"],
                } for r in filtered]), use_container_width=True, hide_index=True)

            # Per-setup reliability
            resolved_recs = [r for r in archive_fresh if r["outcome"] not in ("PENDING","EXPIRED")]
            if resolved_recs:
                st.subheader("Setup reliability breakdown")
                setup_map: dict = {}
                for r in resolved_recs:
                    k = normalize_setup_key(r.get("setup", ""))
                    if k not in setup_map:
                        setup_map[k] = {"wins": 0, "losses": 0, "gains": [], "captures": []}
                    w = r["outcome"] in ("HIT_TP1","HIT_TP2","MANUAL_WIN")
                    setup_map[k]["wins" if w else "losses"] += 1
                    if r["outcome_pct"] is not None:
                        setup_map[k]["gains"].append(r["outcome_pct"])
                    if r["outcome"] == "HIT_TP1" and r.get("expected_pct") and r.get("outcome_pct") is not None:
                        ep = float(r["expected_pct"])
                        if ep > 0:
                            setup_map[k]["captures"].append(r["outcome_pct"] / ep * 100)

                rows = []
                for k, s in setup_map.items():
                    t = s["wins"] + s["losses"]
                    cap = s["captures"]
                    rows.append({
                        "Setup":        k,
                        "Total":        t,
                        "Win rate":     f"{round(s['wins']/t*100,1)}%" if t else "—",
                        "Avg result":   f"{round(sum(s['gains'])/len(s['gains']),1)}%" if s["gains"] else "—",
                        "TP1 capture":  f"{round(sum(cap)/len(cap),1)}%" if cap else "—",
                        "W / L":        f"{s['wins']} / {s['losses']}",
                    })
                rows.sort(key=lambda x: x["Total"], reverse=True)
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    time.sleep(refresh)
