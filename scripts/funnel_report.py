"""
funnel_report.py — summarize signal_funnel.jsonl (see log_funnel_event() in
src/bot/scanner.py for what gets logged and why).

Two event types:
  veto_dropped — a candidate cleared the confidence/RR/expected-move gate
                 (a real, qualifying BUY) but got downgraded by a macro veto
                 (BTC downtrend, BTC dominance rising, funding rate).
  cap_dropped  — a candidate qualified as a BUY but was crowded out because
                 more than 3 qualifying BUYs showed up in the same scan cycle
                 (app.py's top-3-per-cycle cap only archives the top 3 by
                 confidence/R:R).

This exists because the 2026-09-07 investigation found real, confluence-
passing signals that never appeared in trade_archive.json with no visibility
into why — this makes that funnel visible instead of signals silently
vanishing between "detected" and "logged".

Usage:
    python scripts/funnel_report.py
    python scripts/funnel_report.py --since 2026-09-07
"""
import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime

FUNNEL_LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "signal_funnel.jsonl")


def load_events(since: str | None):
    events = []
    if not os.path.exists(FUNNEL_LOG_PATH):
        return events
    cutoff = datetime.fromisoformat(since) if since else None
    with open(FUNNEL_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if cutoff:
                ts = datetime.fromisoformat(rec["ts"].replace("Z", "+00:00")).replace(tzinfo=None)
                if ts < cutoff:
                    continue
            events.append(rec)
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=str, default=None, help="ISO date, e.g. 2026-09-07")
    args = ap.parse_args()

    events = load_events(args.since)
    if not events:
        print("No funnel events logged yet (signal_funnel.jsonl empty or missing).")
        print("This is expected until the instrumented code has run through at least one scan cycle.")
        return

    print(f"Total funnel events: {len(events)}\n")

    by_event = Counter(e["event"] for e in events)
    print("=== By event type ===")
    for ev, n in by_event.most_common():
        print(f"  {ev:15s} {n}")

    veto_dropped = [e for e in events if e["event"] == "veto_dropped"]
    if veto_dropped:
        print(f"\n=== veto_dropped breakdown (N={len(veto_dropped)}) ===")
        veto_counter = Counter()
        for e in veto_dropped:
            for v in e.get("vetoes", []):
                veto_counter[v] += 1
        for v, n in veto_counter.most_common():
            print(f"  {v:20s} {n}")

        print("\n  Most recent 15 veto_dropped events:")
        for e in veto_dropped[-15:]:
            print(f"    {e['ts'][:16]}  {e['symbol']:15s} conf={e.get('conf')} rr={e.get('rr')} "
                  f"vetoes={e.get('vetoes')} setup={str(e.get('setup',''))[:40]}")

    cap_dropped = [e for e in events if e["event"] == "cap_dropped"]
    if cap_dropped:
        print(f"\n=== cap_dropped (N={len(cap_dropped)}) — qualifying BUYs crowded out by top-3 cap ===")
        by_symbol = Counter(e["symbol"] for e in cap_dropped)
        print("  Most frequently crowded-out symbols:")
        for sym, n in by_symbol.most_common(10):
            print(f"    {sym:15s} {n}")
        print("\n  Most recent 15 cap_dropped events:")
        for e in cap_dropped[-15:]:
            print(f"    {e['ts'][:16]}  {e['symbol']:15s} conf={e.get('conf')} rr={e.get('rr')} "
                  f"setup={str(e.get('setup',''))[:40]}")

    print("\n" + "=" * 72)
    print("If cap_dropped dominates: the detection logic is fine, the top-3-per-cycle")
    print("cap in app.py is the bottleneck — consider raising it.")
    print("If veto_dropped dominates: one specific macro veto is suppressing most real")
    print("opportunities — worth checking whether that veto's threshold is too aggressive.")


if __name__ == "__main__":
    main()
