# StockGenie Pro

StockGenie Pro is a Streamlit-based NSE stock scanner for intraday and swing-trading workflows. It analyzes sector-wise watchlists, ranks actionable setups, and displays risk-managed entries with stop loss, target, reward/risk, liquidity, relative strength, RVOL, sector momentum, and trend-quality checks.

> This project is a decision-support tool, not financial advice. Market data can be delayed, unavailable, or incorrect. Always validate signals, liquidity, order execution, and risk before placing trades.

## Features

- NSE sector scanner with single-sector, multi-sector, and all-sector runs.
- Intraday and swing-style recommendations.
- Standard and adaptive recommendation modes.
- Technical indicator scoring using RSI, MACD, Ichimoku Cloud, ATR, VWAP, Bollinger Bands, Donchian Channel, Keltner Channel, OBV, CMF, TRIX, CMO, and related momentum indicators.
- Risk-managed trade levels: current price, buy-at level, stop loss, target, and reward/risk.
- Opportunity ranking based on relative strength, RVOL, sector momentum, liquidity, and entry gap.
- Sector momentum detection to highlight active rotation.
- Major trend conflict protection: blocks bullish top picks when Ichimoku trend is `Strong Sell`.
- Momentum exhaustion penalty: reduces ranking for very hot RVOL setups that are extended from EMA20 or have already made a large daily move.
- SQLite storage for daily picks.
- SmartAPI integration for Angel One market data.

## How The Ranking Works

The opportunity score combines:

- Relative strength versus NIFTY.
- Relative volume participation.
- Sector momentum.
- Liquidity by average traded value.
- Entry quality based on distance from the suggested buy level.
- Exhaustion adjustment for overheated moves.

The scanner favors setups that are liquid, close to entry, backed by sector strength, and not already too extended.

## Important Filters

### Ichimoku Conflict Filter

If Ichimoku trend is `Strong Sell` while other signals show buy strength, the setup is treated as a major trend conflict. Bullish labels are suppressed and the stock is rejected from actionable top picks.

### Momentum Exhaustion Filter

High RVOL can mean institutional participation, but it can also mean a blowoff move. StockGenie Pro applies an adaptive penalty when:

- `RVOL > 5`
- And price is extended beyond the configured daily-move or EMA20-distance threshold.

The penalty scales up for euphoric moves and is capped to avoid overcorrecting.

## Recommended Scan Timing

All timings are in IST.

### Intraday

- 9:20-9:30 AM: first scan, but avoid impulsive entries.
- 10:00-10:30 AM: best primary intraday scan window.
- 12:00-12:30 PM: optional re-scan.
- 2:15-2:45 PM: late momentum scan with tighter risk.
- Avoid fresh intraday entries after 3:00 PM unless execution is very controlled.

### Swing

- 3:15-3:30 PM: best live swing scan while the daily candle is nearly complete.
- 3:45-5:00 PM: best final swing watchlist scan after market close.
- 8:45-9:05 AM next day: review news and gap risk.
- 9:20-10:00 AM: execute only if price remains near the planned buy level.

## Project Structure

```text
.
|-- dd.py                 # Main Streamlit app and scanner logic
|-- sector_perf.py        # Sector performance helper
|-- requirements.txt      # Python dependencies
|-- .env.example          # SmartAPI credential template
|-- stock_picks.db        # Local SQLite picks database
|-- stock_data_cache/     # Cached market data
|-- cache_directory/      # App cache
`-- logs/                 # Runtime logs
```

Cache, log, database, and environment files are local runtime artifacts and should not be used as trading records without validation.

## Setup

### 1. Clone The Repository

```bash
git clone https://github.com/Ashbinbiju/StockGenie-Pro.git
cd StockGenie-Pro
```

### 2. Create A Virtual Environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Credentials

Copy the example environment file:

```bash
cp .env.example .env
```

Fill in your Angel One SmartAPI credentials:

```env
CLIENT_ID=your_angelone_client_id
PASSWORD=your_angelone_password_or_pin
TOTP_SECRET=your_totp_secret
API_KEY=your_smartapi_api_key
HISTORICAL_API_KEY=your_smartapi_api_key
TRADING_API_KEY=your_smartapi_trading_api_key
MARKET_API_KEY=your_smartapi_market_api_key
```

For Streamlit Cloud, configure the same values as Streamlit secrets.

### 5. Run The App

```bash
streamlit run dd.py
```

Open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

## Usage

1. Choose one or more sectors from the sidebar.
2. Select `Standard` or `Adaptive` recommendation mode.
3. Run the all-sector or sector-specific scanner.
4. Review top picks by score, entry gap, reward/risk, liquidity, sector momentum, and audit text.
5. Use the single-stock analysis view for detailed charts and indicator context.

## Deployment

The repository can be deployed to Streamlit Cloud or any Python host that supports Streamlit.

Basic deployment command:

```bash
streamlit run dd.py
```

For hosted deployment, make sure environment variables or Streamlit secrets are configured before running the app.

## Risk Notes

- Do not chase stocks far above the suggested buy level.
- Prefer clean entries with small entry gaps, strong liquidity, and sector confirmation.
- Treat very high RVOL with caution when the stock is already extended.
- Avoid buy setups with major trend conflicts.
- Always define position size, stop loss, and maximum daily loss before entering trades.

## License

No license file is currently included. Add a license before distributing or accepting external contributions.
