from bot.scanner import scan_spot_and_stocks, render_console

CRYPTO = [
    "DOGE/USDT","HIGH/USDT","SHIB/USDT","CFX/USDT","ARPA/USDT","TLM/USDT","STRK/USDT",
    "BTC/USDT","ETH/USDT","ZRO/USDT","XRP/USDT","FIDA/USDT","MOVR/USDT","GALA/USDT",
    "ARB/USDT","ATOM/USDT","JASMY/USDT","TIA/USDT"
]

STOCKS = ["EIMI.L"]

ideas = scan_spot_and_stocks(
    crypto_symbols=CRYPTO,
    stock_tickers=STOCKS,
    timeframe="1h",
    limit=500,
    holdings_path="holdings.json",
)

render_console(ideas, top=20, diag=True)
