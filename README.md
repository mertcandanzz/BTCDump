# BTCDump - Professional Bitcoin Signal Tool

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A professional-grade Bitcoin signal tool using ensemble machine learning (XGBoost, Random Forest, Gradient Boosting) with walk-forward validation, confidence scoring, and backtesting.

## Features

- **Walk-Forward Validated ML**: 5-fold expanding-window cross-validation prevents data leakage
- **Weighted Ensemble**: Inverse-MAPE weighted combination of 3 models
- **Confidence Scoring**: 0-100% composite score (model agreement + indicator confluence)
- **10 Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, Stochastic, ADX, OBV, VWAP, MAs
- **Backtesting Engine**: Win rate, profit factor, Sharpe ratio, max drawdown, equity curve
- **Threshold Optimization**: Grid search for optimal signal thresholds on historical data
- **Model Persistence**: Save/load trained models between sessions (joblib)
- **Smart Auto-Live**: Only retrains when enough new candles arrive
- **Data Caching**: Avoids redundant API calls with configurable TTL

## Installation

```bash
git clone https://github.com/codingcreatively/BTCDump.git
cd BTCDump

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py
# or
python -m btcdump
```

### Menu Options

1. **Select Timeframe** - Choose interval (5m, 30m, 1h, 4h, 1d)
2. **Train & Predict** - Walk-forward training + prediction with confidence score
3. **Show Live Chart** - Candlestick chart with indicators (RSI, MACD, BB, volume)
4. **Run Backtest** - Full walk-forward backtest with performance metrics
5. **Show Last Signal** - Display previous analysis
6. **Signal History** - View all generated signals
7. **Auto Live Mode** - Continuous predictions with smart retraining

## Architecture

```
btcdump/
├── config.py          Dataclass-based configuration (all parameters)
├── utils.py           Logging, retry decorator, path helpers
├── data.py            Binance API fetching, caching, validation
├── indicators.py      10 technical indicators (pure functions)
├── features.py        Sliding-window feature engineering
├── models.py          Walk-forward training, weighted ensemble, persistence
├── signals.py         Signal generation with confidence scoring
├── backtest.py        Walk-forward backtester with threshold optimization
├── visualization.py   Candlestick charts, equity curves
└── app.py             CLI menu and display logic
```

## Signal Generation

Signals combine ML prediction with indicator confluence:

| Component | Weight | Description |
|-----------|--------|-------------|
| Model Spread | 50% | How similar are the 3 model predictions? |
| Directional Agreement | 30% | Do all models agree on direction? |
| Indicator Confluence | 20% | How many of 5 indicators confirm? |

Signals with confidence below 30% are automatically classified as HOLD.

## Technical Indicators

| Indicator | Parameters |
|-----------|-----------|
| Moving Averages | MA5, MA20, MA50 |
| RSI | 14-period (Wilder's smoothing) |
| MACD | 12/26 EMA, 9-period signal |
| Bollinger Bands | 20-period, 2 std dev |
| ATR | 14-period |
| Stochastic | %K=14, %D=3 |
| ADX | 14-period |
| OBV | Z-score normalized |
| VWAP | 20-period rolling proxy |
| Volume Ratio | Volume / 20-period SMA |

## Backtest Metrics

- Win Rate, Profit Factor, Max Drawdown
- Sharpe Ratio (annualized)
- Per-signal-type accuracy
- Equity curve with drawdown visualization
- Optimal threshold discovery via grid search

## Web UI with Multi-LLM Integration

BTCDump includes a web interface with real-time multi-AI discussion capabilities.

```bash
python main.py --web          # Start web UI at http://localhost:8000
python main.py --web --port 3000  # Custom port
python run_web.py             # Shortcut
```

### Supported LLM Providers

| Provider | Models | API Key Env Var |
|----------|--------|-----------------|
| OpenAI | GPT-4o, GPT-4o-mini, GPT-4-turbo, o3-mini | `OPENAI_API_KEY` |
| Claude | Opus 4.6, Sonnet 4.6, Haiku 4.5 | `ANTHROPIC_API_KEY` |
| Grok | Grok-3, Grok-3-mini, Grok-3-fast | `XAI_API_KEY` |
| Gemini | Gemini 2.5 Pro, 2.5 Flash, 2.0 Flash | `GOOGLE_API_KEY` |

### Features

- **AI Discussion Panel**: All active models debate your questions in multi-round discussions
- **Individual Chat Windows**: Chat privately with each model about the current market state
- **Signal Dashboard**: Real-time signal display with all indicators
- **Settings UI**: Configure API keys and select models per provider

### Discussion Mode

Ask a question and all active AI models will:
1. Answer independently (Round 1)
2. Read each other's answers and respond (Round 2)
3. Final thoughts with potential mind-changes (Round 3)

Models reference each other by name, agree/disagree, and produce a natural debate.

## Disclaimer

**This tool is for educational and research purposes only.**

- Cryptocurrency trading involves significant risk
- Past performance does not guarantee future results
- Do not use this tool for actual trading decisions
- The author assumes no liability for financial losses
