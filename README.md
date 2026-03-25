# BTCDump v5.1 Pro

Professional-grade crypto signal platform with 124 ML features, walk-forward validation, ensemble stacking, and a complete trading workflow.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Features](https://img.shields.io/badge/ML%20Features-124-brightgreen)
![Routes](https://img.shields.io/badge/API%20Routes-110-blue)
![Tests](https://img.shields.io/badge/Tests-57%20passing-green)

## Quick Start

```bash
pip install -r requirements.txt
python run_web.py
# Open http://localhost:8000
```

**Docker:**
```bash
docker build -t btcdump .
docker run -p 8000:8000 btcdump
```

## Platform Overview

| Metric | Value |
|--------|-------|
| ML Features | 124 (2,480 dimensions) |
| API Routes | 110 |
| Code Lines | 13,500+ |
| UI Modes | Dashboard, Analysis, Compare |
| AI Providers | OpenAI, Claude, Grok, Gemini |
| Tests | 57 passing |

## ML Pipeline

```
Raw OHLCV --> 124 Features --> StandardScaler --> L1 Selection (LassoCV)
                                                       |
                                               +-------+-------+
                                               |       |       |
                                            XGBoost    RF   GradBoost
                                               |       |       |
                                               +---+---+---+---+
                                                   |
                                             Ridge Meta-Learner (Stacking)
                                                   |
                                             Final Prediction
                                                   |
                                         8-Component Consensus Engine
                                                   |
                                          0-100 Conviction Score
```

### 124 Features across 29 Categories

Classic TA (17) | Returns & Momentum (12) | Volatility (11) | Candlestick Body (4) | Bollinger & Squeeze (4) | Volume & Liquidity (6) | Trend (4) | Regime Detection (6) | Time Cyclical (4) | Distribution (2) | Ichimoku Cloud (5) | Volume Profile (2) | Pivot Points (4) | Order Flow (2) | Candlestick Patterns (7) | Microstructure (5) | Statistical (6) | Whale Detection (2) | Shannon Entropy (1) | Adaptive MAs (4) | Seasonality (2) | FFT Cycle Detection (3) | Transfer Entropy (2) | Approximate Entropy (2) | DFA (1) | Feature Interactions (5) | Jump Detection (3) | Kalman Filter (2) | Core OHLCV (2)

## Key Features

### Signal Intelligence
- **Consensus Engine**: 8 components (ML, technicals, volume, regime, patterns, momentum, seasonality, health) into one 0-100 score
- **Direction Probability**: UP/DOWN % from ensemble disagreement
- **Prediction Intervals**: 68% and 95% confidence ranges
- **Regime-Adaptive Thresholds**: auto-adjust for market conditions
- **Multi-TF Confluence**: weighted 15m/1h/4h/1d score

### Chart
- TradingView-quality candlestick + Heikin-Ashi toggle
- EMA, Ichimoku Cloud, S/R Levels, Fibonacci, Auto Trend Lines
- Zoom, Pan, Crosshair, Ruler tool

### Trading
- Paper trading with SL/TP and trade journal
- Trade Setup Generator (A-D grade, 5-point checklist)
- Kelly Criterion + conviction-based sizing
- DCA Simulator, Portfolio Optimizer (Markowitz)
- Execution Cost Estimator
- AI Trade Coach (personalized advice)

### Analysis
- Market Scanner (9 conditions)
- 5-Strategy Backtester with realistic fees
- Strategy vs Buy-and-Hold comparison
- Monte Carlo Simulation
- Seasonality, Pair Trading, Correlation Breakdown
- Feature Drift Detection, Signal Calibration

### Compare Mode (5 tabs)
Grid | Heatmap | Correlation | Scanner | Leaderboard + Momentum Rotation

### Integration
- Binance WebSocket live feed
- Telegram + Discord notifications
- Webhook API + SSE Stream
- CSV Export | 4-provider AI Chat

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `D` | Dashboard | `S` | Analysis | `C` | Compare |
| `B` | Buy | `Shift+S` | Short | `X` | Close | `T` | Trade Setup |
| `R` | Refresh | `Ctrl+K` | Search |

## Testing

```bash
python -m pytest tests/ -v
```

## API Docs

Swagger UI at `http://localhost:8000/docs` when server is running.

## Disclaimer

**Educational and research purposes only.** Cryptocurrency trading involves significant risk. Do not use for actual trading decisions.
