# 🧠 AI Brain Pro - Professional Trading Dashboard

![Version](https://img.shields.io/badge/version-3.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red)
![License](https://img.shields.io/badge/license-MIT-green)

A Bloomberg-style professional trading dashboard with AI-powered analysis, real-time signals, and comprehensive backtesting capabilities.

## ✨ Features

### 🎨 Professional UI
- **Bloomberg Terminal Style** - Dark mode aesthetic with glassmorphism design
- **Real-time KPI Cards** - Portfolio value, P&L, Win Rate with color-coded indicators
- **Interactive Charts** - Plotly-powered candlestick charts with SMA overlays

### 📊 Dashboard
- Live trading signals for 20+ major markets
- Technical analysis with RSI, MACD, Bollinger Bands
- AI-generated strategy insights

### 🔮 AI Prediction
- Price predictions with confidence bands
- Mean reversion logic
- Volatility-adjusted forecasting

### 📈 Backtesting
- Historical strategy performance
- Equity curve visualization
- Trade-by-trade analysis
- Win rate and profit factor metrics

### 💹 Paper Trading
- $100k virtual capital
- Open positions tracking
- Trade history with P&L

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/quitsaurabhverma2008-sketch/ai-brain.git
cd ai-brain

# Install dependencies
pip install -r requirements.txt

# Run the application
cd src/trading_brain
streamlit run app.py --server.port 8514
```

Visit: **http://localhost:8514**

## 📁 Project Structure

```
ai-brain/
├── src/
│   └── trading_brain/
│       ├── app.py              # Main Pro Dashboard
│       ├── dashboard.py        # Original Dashboard
│       ├── prediction.py       # AI Prediction
│       ├── trade_now.py        # Paper Trading
│       ├── data_feed.py        # Yahoo Finance Data
│       ├── indicators.py       # Technical Indicators
│       ├── signals.py          # Signal Generation
│       ├── trader.py           # Portfolio Management
│       └── ml_model.py         # ML Predictions
├── requirements.txt
├── README.md
└── LICENSE
```

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Frontend** | Streamlit, Plotly |
| **ML** | XGBoost, Scikit-learn |
| **Data** | Yahoo Finance (yfinance) |
| **Analysis** | Pandas, NumPy |
| **API** | Requests |

## 📊 Technical Indicators

- **Trend**: SMA 20/50, EMA
- **Momentum**: RSI, MACD, Stochastic, CCI
- **Volatility**: Bollinger Bands, ATR
- **Volume**: OBV, Volume SMA

## ⚠️ Disclaimer

**This software is for educational and informational purposes only.**

- Not financial advice
- Past performance does not guarantee future results
- Always do your own research before trading
- The developers are not responsible for any financial losses

---

**Built with ❤️ by [Saurabh](https://github.com/quitsaurabhverma2008-sketch)**

*© 2024 AI Brain Pro. All rights reserved.*
