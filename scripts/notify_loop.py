import json
import os
import time
import smtplib
import ssl
import subprocess
from datetime import datetime, timezone
from email.mime.text import MIMEText

from bot.scanner import scan_spot_and_stocks, top_binance_spot_symbols, DEFAULT_STOCKS, load_holdings

# ============================================================
# SETTINGS — YOU CAN EDIT THESE SAFELY
# ============================================================

HOLDINGS_PATH = "holdings.json"

CRYPTO_SYMBOLS = [
    "DOGE/USDT","HIGH/USDT","SHIB/USDT","CFX/USDT","ARPA/USDT","TLM/USDT","STRK/USDT",
    "BTC/USDT","ETH/USDT","ZRO/USDT","XRP/USDT","FIDA/USDT","MOVR/USDT","GALA/USDT",
    "ARB/USDT","ATOM/USDT","JASMY/USDT","TIA/USDT"
]

# Set to True to auto-scan the most liquid USDT spot pairs (overrides CRYPTO_SYMBOLS)
AUTO_TOP_BINANCE = True
TOP_LIMIT = 60

STOCK_TICKERS = ["EIMI.L"]

# Set to True to use a preset US large-cap basket for stocks/ETFs
AUTO_STOCK_BASKET = True
STOCK_BASKET = DEFAULT_STOCKS

CRYPTO_TIMEFRAME = "1h"
STOCK_TIMEFRAME = "1d"
CANDLES = 500

# Scan frequency
CRYPTO_SCAN_EVERY_SECONDS = 15 * 60     # 15 minutes
STOCK_SCAN_EVERY_SECONDS = 6 * 60 * 60  # 6 hours

# Alert thresholds
CONFIDENCE_BUY = 70
CONFIDENCE_SELL = 65

STATE_FILE = ".alert_state.json"

# Load holdings to filter out owned assets from opportunity alerts
HOLDINGS = load_holdings(HOLDINGS_PATH)
OWN_CRYPTO = set(HOLDINGS.get("crypto", {}).keys())
OWN_STOCKS = set(HOLDINGS.get("stocks", {}).keys())

# Email settings (set via environment variables)
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
ALERT_EMAIL_TO = [e.strip() for e in os.getenv("ALERT_EMAIL_TO", "").split(",") if e.strip()]
EMAIL_ENABLED = all([SMTP_HOST, SMTP_USER, SMTP_PASS, ALERT_EMAIL_TO])

# ============================================================
# INTERNAL HELPERS
# ============================================================

def mac_notify(title: str, message: str):
    """Native macOS notification (Apple Silicon safe)."""
    script = f'display notification "{message}" with title "{title}"'
    subprocess.run(["osascript", "-e", script])

def send_email(subject: str, body: str):
    if not EMAIL_ENABLED:
        return
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = ", ".join(ALERT_EMAIL_TO)

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.starttls(context=context)
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, ALERT_EMAIL_TO, msg.as_string())
    except Exception as e:
        print(f"✉️ Email send failed: {e}")


def load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {
            "last": {},
            "last_crypto_scan": 0,
            "last_stock_scan": 0
        }
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state: dict):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def format_idea(i) -> str:
    entry = "-" if not i.entry else f"{i.entry[0]:.4f} – {i.entry[1]:.4f}"
    stop = "-" if i.stop is None else f"{i.stop:.4f}"
    targets = "-" if not i.targets else ", ".join(f"{t:.4f}" for t in i.targets)
    return (
        f"{i.market} {i.symbol}\n"
        f"Action: {i.action}\n"
        f"Confidence: {i.confidence}%\n"
        f"Entry: {entry}\n"
        f"Stop: {stop}\n"
        f"Targets: {targets}"
    )


def should_alert(i) -> bool:
    if i.action == "BUY":
        return i.confidence >= CONFIDENCE_BUY
    if i.action == "SELL":
        return i.confidence >= CONFIDENCE_SELL
    return False


def handle_alerts(ideas, state):
    for i in ideas:
        if not should_alert(i):
            continue

        # Skip alerts for assets already owned
        if i.market == "CRYPTO" and i.symbol in OWN_CRYPTO:
            continue
        if i.market == "STOCK" and i.symbol in OWN_STOCKS:
            continue

        key = f"{i.market}:{i.symbol}"
        signature = f"{i.action}:{i.confidence}:{i.entry}:{i.stop}:{i.targets}"

        if state["last"].get(key) != signature:
            mac_notify("Portfolio Advisor", format_idea(i))
            state["last"][key] = signature
            subject = f"Portfolio Advisor: {i.market} {i.symbol} → {i.action}"
            body = format_idea(i)
            send_email(subject, body)


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    state = load_state()

    print("🚀 Portfolio Advisor Notifier started")
    print("🔁 Running continuously (Ctrl+C to stop)")
    print()

    while True:
        now = time.time()

        # Refresh dynamic crypto universe if requested
        if AUTO_TOP_BINANCE:
            try:
                crypto_list = top_binance_spot_symbols(limit=TOP_LIMIT)
            except Exception as e:
                print(f"⚠️ Could not load top markets, using static list. Error: {e}")
                crypto_list = CRYPTO_SYMBOLS
        else:
            crypto_list = CRYPTO_SYMBOLS

        # Pick stock universe
        stock_list = STOCK_BASKET if AUTO_STOCK_BASKET else STOCK_TICKERS

        # ---------------- CRYPTO SCAN ----------------
        if now - state["last_crypto_scan"] >= CRYPTO_SCAN_EVERY_SECONDS:
            print(f"[{datetime.now(timezone.utc).isoformat()}] Crypto scan…")

            ideas = scan_spot_and_stocks(
                crypto_symbols=crypto_list,
                stock_tickers=[],
                timeframe=CRYPTO_TIMEFRAME,
                limit=CANDLES,
                holdings_path=HOLDINGS_PATH,
            )

            handle_alerts(ideas, state)
            state["last_crypto_scan"] = now
            save_state(state)

        # ---------------- STOCK SCAN ----------------
        if now - state["last_stock_scan"] >= STOCK_SCAN_EVERY_SECONDS:
            print(f"[{datetime.now(timezone.utc).isoformat()}] Stock/ETF scan…")

            ideas = scan_spot_and_stocks(
                crypto_symbols=[],
                stock_tickers=stock_list,
                timeframe=STOCK_TIMEFRAME,
                limit=CANDLES,
                holdings_path=HOLDINGS_PATH,
            )

            handle_alerts(ideas, state)
            state["last_stock_scan"] = now
            save_state(state)

        time.sleep(5)


if __name__ == "__main__":
    main()
