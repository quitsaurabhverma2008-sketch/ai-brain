# 🧠 AI Brain - Intelligent Trading System

![Version](https://img.shields.io/badge/version-2.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red)

AI-powered stock trading system with real-time signals, price predictions, and paper trading.

## ✨ Features

- **📊 Live Dashboard** - Real-time AI trading signals for 200+ markets
- **🔮 Price Prediction** - AI-powered future price predictions with confidence bands
- **🎯 Paper Trading** - Practice trading with $100k virtual capital
- **🤖 ML Model** - XGBoost classifier trained on 37,000+ samples
- **📈 Technical Indicators** - RSI, MACD, SMA, Bollinger Bands, Stochastic, ADX, CCI, ATR, VWAP

## 🚀 Getting Started

```bash
# Install dependencies
pip install streamlit pandas numpy plotly yfinance xgboost scikit-learn

# Run the app
cd src/trading_brain
streamlit run dashboard.py --server.port 8514
```

## 🌐 Live Demo

Visit: http://localhost:8514

## 📱 Pages

1. **Dashboard** - Live signals with technical analysis charts
2. **Prediction** - AI price predictions with future candles
3. **Trade Now** - Paper trading simulation

## 🛠️ Tech Stack

- **Frontend**: Streamlit, Plotly
- **ML**: XGBoost, Scikit-learn
- **Data**: Yahoo Finance (yfinance)
- **Analysis**: Pandas, NumPy

## ⚠️ Disclaimer

This project is for educational purposes only. Not financial advice.

---

**Built with ❤️ by Saurabh**
