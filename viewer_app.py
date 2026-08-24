"""
viewer_app.py — read-only dashboard for Streamlit Community Cloud.

Displays trade_archive.json (kept in sync from the local machine, where the actual
scanning bot runs continuously — see scripts/sync_archive.py). This page does NOT
scan, does NOT hold API keys, and does NOT write back to the archive: Community
Cloud's filesystem is ephemeral and separate from the local bot's, so any edit made
here would just be lost on the next restart. For the full interactive app (manual
outcome resolution, notes, live scanning), run app.py locally.
"""
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath("src"))

from bot.archive_store import (  # noqa: E402
    load_archive,
    normalize_setup_key,
    outcome_badge,
    reliability_stats,
)

st.set_page_config(page_title="Trading Signals (Read-only)", layout="wide")
st.title("Trading Signals — Read-only Mirror")
st.caption(
    "Mirrors the archive from the bot running locally. This page never scans and "
    "never writes back — for the live/interactive app, run app.py locally."
)

archive = load_archive()

if not archive:
    st.info(
        "No signals in the archive yet. If this is a fresh deploy, the local bot's "
        "sync job hasn't pushed trade_archive.json yet — see scripts/sync_archive.py."
    )
    st.stop()

if os.path.exists("trade_archive.json"):
    synced_at = pd.Timestamp(os.path.getmtime("trade_archive.json"), unit="s", tz="UTC")
    st.caption(f"Archive last updated: {synced_at.strftime('%Y-%m-%d %H:%M UTC')}")

stats = reliability_stats(archive)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total signals", stats["total"])
c2.metric("Resolved", stats["resolved"])
c3.metric("Win rate", f"{stats['win_rate']}%")
c4.metric("Avg result", f"{stats['avg_gain']}%")
c5.metric("Pending", stats["pending"])

st.divider()
st.subheader("Signal Archive")

filter_out = st.multiselect(
    "Filter by outcome",
    ["PENDING", "HIT_TP1", "HIT_TP2", "STOPPED_OUT", "EXPIRED", "MANUAL_WIN", "MANUAL_LOSS"],
    default=["PENDING", "HIT_TP1", "HIT_TP2", "STOPPED_OUT"],
)
filtered = [r for r in reversed(archive) if r["outcome"] in filter_out]

if not filtered:
    st.info("No records match filters.")
else:
    st.dataframe(pd.DataFrame([{
        "ID":        r["id"],
        "Logged":    r["logged_at"][:16].replace("T", " "),
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
        "Resolved":  (r["resolved_at"] or "")[:16].replace("T", " "),
        "Setup":     r["setup"][:45] if r["setup"] else "",
        "Notes":     r["notes"],
    } for r in filtered]), use_container_width=True, hide_index=True)

resolved_recs = [r for r in archive if r["outcome"] not in ("PENDING", "EXPIRED")]
if resolved_recs:
    st.subheader("Setup reliability breakdown")
    setup_map: dict = {}
    for r in resolved_recs:
        k = normalize_setup_key(r.get("setup", ""))
        if k not in setup_map:
            setup_map[k] = {"wins": 0, "losses": 0, "gains": [], "captures": []}
        w = r["outcome"] in ("HIT_TP1", "HIT_TP2", "MANUAL_WIN")
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
            "Setup":       k,
            "Total":       t,
            "Win rate":    f"{round(s['wins'] / t * 100, 1)}%" if t else "—",
            "Avg result":  f"{round(sum(s['gains']) / len(s['gains']), 1)}%" if s["gains"] else "—",
            "TP1 capture": f"{round(sum(cap) / len(cap), 1)}%" if cap else "—",
            "W / L":       f"{s['wins']} / {s['losses']}",
        })
    rows.sort(key=lambda x: x["Total"], reverse=True)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
