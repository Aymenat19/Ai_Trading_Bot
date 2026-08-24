from __future__ import annotations

import argparse
from typing import List, Optional

from bot.scanner import run_scan, DEFAULT_SYMBOLS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Read-only Binance trading advisor (no orders).")
    p.add_argument("--timeframe", default="1h", help="Timeframe like 15m, 1h, 4h, 1d")
    p.add_argument("--limit", type=int, default=500, help="Number of candles to fetch")
    p.add_argument("--top", type=int, default=5, help="How many top rows to show in table")
    p.add_argument(
        "--symbols",
        default="",
        help="Comma-separated symbols (e.g. BTC/USDT,ETH/USDT). Leave empty for defaults.",
    )
    p.add_argument("--no-diag", action="store_true", help="Hide per-symbol diagnostics")
    return p.parse_args()


def main():
    args = parse_args()

    symbols: Optional[List[str]] = None
    if args.symbols.strip():
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = DEFAULT_SYMBOLS

    run_scan(
        symbols=symbols,
        timeframe=args.timeframe,
        limit=args.limit,
        top=args.top,
        diag=not args.no_diag,
    )


if __name__ == "__main__":
    main()
