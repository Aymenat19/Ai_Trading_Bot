AI Trading Bot (spot-only advisor)
---------------------------------

Quick start
- Create a virtualenv and install deps: `pip install pandas numpy requests python-dotenv rich ccxt yfinance streamlit`
- Launch Streamlit UI: `PYTHONPATH=src streamlit run app.py`
- Run console scanner: `PYTHONPATH=src python -m bot.main --timeframe 1h --top 10`
- Start macOS notifications loop: `PYTHONPATH=src python scripts/notify_loop.py`

Notes
- Holdings live in `holdings.json` and drive HOLD vs BUY/SELL suggestions.
- The app scans Binance spot symbols; stocks/ETF support uses Yahoo Finance (read-only). UI/notifier offer preset large-cap stock basket toggles; disable to use your own list.
