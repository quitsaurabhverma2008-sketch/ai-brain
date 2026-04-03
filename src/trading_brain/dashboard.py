"""
Trading Brain - Live Dashboard V2
==================================
Complete trading dashboard with real-time signals, portfolio tracking, and charts
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time
import sys
import os

# Add parent directory to path
import sys
import os
src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_dir)

from data_feed import DataFeed
from indicators import calculate_all_indicators
from signals import SignalGenerator, ExitSignalGenerator, Signal
from trader import Portfolio, PositionType
from data_collector import STOCK_LISTS


# Page config
st.set_page_config(
    page_title="Trading Brain",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS - Modern Dark Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main { 
        background-color: #0a0a0f;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d14 0%, #12121a 100%);
        border-right: 1px solid #1e1e2e;
    }
    
    /* Modern Cards */
    .stMetric { 
        background: linear-gradient(135deg, #1a1a24 0%, #16161e 100%);
        padding: 20px; 
        border-radius: 16px;
        border: 1px solid #2a2a3a;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    /* Signal Boxes */
    .buy-signal { 
        background: linear-gradient(135deg, #00d4aa 0%, #00a878 100%); 
        color: white; 
        padding: 15px 25px; 
        border-radius: 12px; 
        text-align: center;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(0, 212, 170, 0.3);
    }
    .sell-signal { 
        background: linear-gradient(135deg, #ff4757 0%, #c0392b 100%); 
        color: white; 
        padding: 15px 25px; 
        border-radius: 12px; 
        text-align: center;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(255, 71, 87, 0.3);
    }
    .hold-signal { 
        background: linear-gradient(135deg, #5a5a7a 0%, #3d3d5c 100%); 
        color: white; 
        padding: 15px 25px; 
        border-radius: 12px; 
        text-align: center;
        font-weight: 600;
    }
    
    .positive { color: #00d4aa; }
    .negative { color: #ff4757; }
    
    /* Headers */
    h1, h2, h3 {
        background: linear-gradient(90deg, #00d4aa, #00a878);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00d4aa 0%, #00a878 100%);
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        color: white;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 212, 170, 0.4);
    }
    
    /* Radio buttons */
    [data-testid="stRadio"] > div {
        background: #1a1a24;
        border-radius: 12px;
        padding: 10px;
        border: 1px solid #2a2a3a;
    }
    
    /* Dataframes */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        border: 1px solid #2a2a3a;
    }
    
    /* Tabs */
    .stTabs [data-testid="stTabBar"] {
        background: #1a1a24;
        border-radius: 12px;
        padding: 8px;
    }
    
    /* Expander */
    [data-testid="stExpander"] {
        background: #1a1a24;
        border-radius: 12px;
        border: 1px solid #2a2a3a;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #1a1a24;
    }
    ::-webkit-scrollbar-thumb {
        background: #3a3a4a;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #4a4a5a;
    }
</style>
""", unsafe_allow_html=True)


class TradingBrain:
    """Main trading brain controller"""
    
    def __init__(self):
        self.data_feed = DataFeed()
        self.signal_gen = SignalGenerator()
        self.exit_gen = ExitSignalGenerator()
        self.portfolio = Portfolio(100000)
        
        # All markets - default to just major ones for speed
        self.default_stocks = [
            # Top US Stocks (50)
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'JPM', 'JNJ',
            'V', 'PG', 'UNH', 'MA', 'HD', 'DIS', 'PYPL', 'BAC', 'ADBE', 'CRM',
            'NFLX', 'INTC', 'VZ', 'T', 'XOM', 'KO', 'PEP', 'WMT', 'ABT', 'MRK',
            'CVX', 'LLY', 'PFE', 'ABBV', 'TMO', 'COST', 'AVGO', 'NEE', 'DHR', 'NKE',
            'TXN', 'QCOM', 'HON', 'UPS', 'PM', 'MS', 'GS', 'BLK', 'IBM', 'AMD',
            # Top Crypto (10)
            'BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 
            'AVAX-USD', 'DOGE-USD', 'DOT-USD', 'MATIC-USD',
            # Top ETFs (5)
            'SPY', 'QQQ', 'IWM', 'GLD', 'TLT'
        ]
    
    def analyze_stock(self, symbol: str, interval: str = "1h", period: str = "30d") -> dict:
        """Analyze a single stock and return signal data."""
        df = self.data_feed.get_intraday_data(symbol, interval, period)
        
        if df.empty or len(df) < 50:
            return {'error': 'Insufficient data', 'rows': len(df) if df is not None else 0}
        
        # Calculate indicators
        df = calculate_all_indicators(df)
        
        # Generate signal
        signal, score, details = self.signal_gen.generate_signal(df)
        
        # Get latest values
        latest = df.iloc[-1]
        
        # Get quote
        quote = self.data_feed.get_live_quote(symbol)
        
        return {
            'symbol': symbol,
            'signal': signal.value,
            'score': score,
            'price': latest['Close'],
            'rsi': latest['RSI'],
            'macd': latest['MACD'],
            'macd_signal': latest['MACD_Signal'],
            'sma_20': latest['SMA_20'],
            'sma_50': latest['SMA_50'],
            'bb_upper': latest['BB_Upper'],
            'bb_lower': latest['BB_Lower'],
            'trend': details.get('trend', 'N/A'),
            'volume': latest['Volume'],
            'quote': quote,
            'details': details
        }
    
    def analyze_all(self, symbols: list) -> list:
        """Analyze top stocks only."""
        results = []
        
        # Only analyze first 10 stocks
        for symbol in symbols[:10]:
            try:
                result = self.analyze_stock(symbol)
                results.append(result)
            except Exception as e:
                print(f"Error: {symbol}: {e}")
        
        return results
    
    def get_entry_levels(self, symbol: str, price: float) -> dict:
        """Calculate entry levels for a signal."""
        return self.signal_gen.get_entry_levels(price)


def create_price_chart(df, symbol: str, entry_price: float = None, 
                     sl: float = None, tp: float = None) -> go.Figure:
    """Create interactive price chart with indicators."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name='Price'
    ), row=1, col=1)
    
    # SMA 20
    if 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA_20'], name='SMA 20',
            line=dict(color='#00cc66', width=1)
        ), row=1, col=1)
    
    # SMA 50
    if 'SMA_50' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA_50'], name='SMA 50',
            line=dict(color='#ff6b6b', width=1)
        ), row=1, col=1)
    
    # Bollinger Bands
    if 'BB_Upper' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_Upper'], name='BB Upper',
            line=dict(color='#666', width=1), showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_Lower'], name='BB Lower',
            line=dict(color='#666', width=1), fill='tonexty',
            fillcolor='rgba(255,255,255,0.05)', showlegend=False
        ), row=1, col=1)
    
    # Entry marker
    if entry_price:
        fig.add_hline(y=entry_price, line_dash="dash", 
                     line_color="#00cc66", annotation_text=f"Entry ${entry_price:.2f}")
    
    # SL/TP lines
    if sl:
        fig.add_hline(y=sl, line_dash="dot", 
                     line_color="#ff4b4b", annotation_text=f"SL ${sl:.2f}")
    if tp:
        fig.add_hline(y=tp, line_dash="dot", 
                     line_color="#00cc66", annotation_text=f"TP ${tp:.2f}")
    
    # RSI
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['RSI'], name='RSI',
            line=dict(color='#ffa500', width=2)
        ), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#ff4b4b", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#00cc66", row=2, col=1)
    
    # MACD
    if 'MACD' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MACD'], name='MACD',
            line=dict(color='#00ccff', width=2)
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MACD_Signal'], name='Signal',
            line=dict(color='#ffa500', width=2)
        ), row=3, col=1)
    
    fig.update_layout(
        template='plotly_dark', height=600,
        xaxis_rangeslider_visible=False,
        title=f'{symbol} - Technical Analysis'
    )
    
    return fig


def main():
    # Modern Sidebar with Navigation
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="margin: 0; font-size: 32px;">🧠</h1>
            <h2 style="margin: 5px 0; font-size: 24px; background: linear-gradient(90deg, #00d4aa, #00a878); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                AI Brain
            </h2>
            <p style="color: #888; font-size: 12px; margin: 0;">AI-Powered Trading System</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        st.markdown("### 📍 Navigation")
        query_params = st.query_params
        default_page = query_params.get("page", "Dashboard")
        
        page = st.radio(
            "Go to",
            ["Dashboard", "Prediction", "Trade Now"],
            index=0 if default_page == "Dashboard" else 1 if default_page == "Prediction" else 2,
            label_visibility="collapsed"
        )
        
        if page != default_page:
            st.query_params["page"] = page
    
    # Page routing
    if page == "Prediction":
        from prediction import show_prediction_page
        show_prediction_page()
        return
    
    if page == "Trade Now":
        from trade_now import show_trade_page
        show_trade_page()
        return
    
    # Import brain
    if 'brain' not in st.session_state:
        st.session_state.brain = TradingBrain()
    
    brain = st.session_state.brain
    
    # Main Header
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 20px 0;">
        <div>
            <h1 style="margin: 0;">📊 Live Dashboard</h1>
            <p style="color: #888; margin: 5px 0;">Real-time AI trading signals</p>
        </div>
        <div style="text-align: right;">
            <p style="color: #00d4aa; font-size: 12px; margin: 0;">Built by</p>
            <p style="color: #fff; font-size: 16px; font-weight: 600; margin: 0;">Saurabh</p>
        </div>
    </div>
    <hr style="border: 1px solid #2a2a3a; margin: 20px 0;">
    """, unsafe_allow_html=True)
    
    # Stock selection - All 200+ markets
    all_stocks = brain.default_stocks
    
    selected_stocks = st.sidebar.multiselect(
        "Select Stocks", all_stocks, default=brain.default_stocks
    )
    
    # Timeframe
    interval = st.sidebar.selectbox(
        "Timeframe", ['15m', '1h', '4h', '1d'], index=1
    )
    
    # Capital
    initial_capital = st.sidebar.number_input(
        "Initial Capital ($)", value=100000, step=10000
    )
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=True)
    
    st.sidebar.markdown("---")
    
    # Portfolio stats
    portfolio = brain.portfolio
    st.sidebar.markdown("### 💼 Portfolio")
    st.sidebar.metric("Capital", f"${portfolio.get_total_value():,.0f}")
    st.sidebar.metric("Open Positions", len(portfolio.positions))
    st.sidebar.metric("Unrealized P&L", f"${portfolio.get_unrealized_pnl():,.0f}")
    
    # Main content
    st.title("🧠 Trading Brain - Live Dashboard")
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not selected_stocks:
        st.warning("Select stocks from sidebar")
        return
    
    # Analyze stocks
    with st.spinner("Analyzing stocks..."):
        results = brain.analyze_all(selected_stocks)
    
    # Filter successful results
    valid_results = [r for r in results if 'error' not in r]
    
    # ========================
    # SIGNAL SUMMARY
    # ========================
    st.markdown("## 🎯 Trading Signals")
    
    # Create signal table
    signal_data = []
    for r in valid_results:
        quote = r.get('quote', {})
        change_pct = quote.get('change_pct', 0) if quote else 0
        
        signal_data.append({
            'Symbol': r['symbol'],
            'Signal': r['signal'],
            'Score': r['score'],
            'Price': r['price'],
            'Change %': f"{change_pct:+.2f}%",
            'RSI': f"{r['rsi']:.1f}",
            'Trend': r['trend']
        })
    
    if signal_data:
        df_signals = pd.DataFrame(signal_data)
        
        # Color code signals
        def color_signal(s):
            if s == 'BUY': return 'background-color: #00cc66; color: white'
            if s == 'SELL': return 'background-color: #ff4b4b; color: white'
            return ''
        
        try:
            styled_df = df_signals.style.map(color_signal, subset=['Signal'])
        except:
            styled_df = df_signals
        
        st.dataframe(styled_df, use_container_width=True)
    
    # ========================
    # BUY SIGNALS
    # ========================
    buy_signals = [r for r in valid_results if r['signal'] == 'BUY']
    
    if buy_signals:
        st.markdown("### ✅ BUY Signals")
        
        for r in buy_signals[:5]:  # Only first 5
            st.markdown(f"### 📈 {r['symbol']}")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Price", f"${r['price']:.2f}")
            with col2:
                st.metric("RSI", f"{r['rsi']:.1f}")
            with col3:
                st.metric("Trend", r['trend'])
            with col4:
                sl_price = r['price'] * 0.98
                tp_price = r['price'] * 1.06
                st.metric("SL/TP", f"{sl_price:.0f} / {tp_price:.0f}")
            
            # Chart
            try:
                df = brain.data_feed.get_intraday_data(r['symbol'], interval, "5d")
                if len(df) > 20:
                    df = calculate_all_indicators(df)
                    fig = create_price_chart(df, r['symbol'], r['price'], sl_price, tp_price)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Chart error: {e}")
            
            # Button
            if st.button(f"✅ BUY {r['symbol']}", key=f"buy_{r['symbol']}"):
                opened = portfolio.open_position(
                    stock=r['symbol'],
                    entry_price=r['price'],
                    position_type=PositionType.LONG,
                    stop_loss=sl_price,
                    take_profit=tp_price
                )
                if opened:
                    st.success(f"Opened: {r['symbol']}")
            
            st.markdown("---")
    
    # ========================
    # SELL SIGNALS  
    sell_signals = [r for r in valid_results if r['signal'] == 'SELL']
    
    if sell_signals:
        st.markdown("### ❌ SELL Signals")
        
        for r in sell_signals[:3]:  # Only first 3
            st.markdown(f"### 📉 {r['symbol']}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Price", f"${r['price']:.2f}")
            with col2:
                st.metric("RSI", f"{r['rsi']:.1f}")
            with col3:
                st.metric("Trend", r['trend'])
            st.markdown("---")
    
    # ========================
    # HOLD SIGNALS
    # ========================
    hold_signals = [r for r in valid_results if r['signal'] == 'HOLD']
    
    if hold_signals:
        st.markdown("### ⏸️ HOLD Signals")
        
        hold_df = pd.DataFrame([{
            'Symbol': r['symbol'],
            'Price': f"${r['price']:.2f}",
            'RSI': f"{r['rsi']:.1f}",
            'Score': r['score']
        } for r in hold_signals])
        
        st.dataframe(hold_df, use_container_width=True)
    
    # ========================
    # OPEN POSITIONS
    # ========================
    st.markdown("---")
    st.markdown("## 💼 Open Positions")
    
    if portfolio.positions:
        position_data = []
        for stock, pos in portfolio.positions.items():
            position_data.append({
                'Symbol': stock,
                'Type': pos.position_type.value,
                'Entry': f"${pos.entry_price:.2f}",
                'Current': f"${pos.current_price:.2f}",
                'SL': f"${pos.stop_loss:.2f}",
                'TP': f"${pos.take_profit:.2f}",
                'P&L': f"${pos.unrealized_pnl:.2f}"
            })
        
        st.dataframe(pd.DataFrame(position_data), use_container_width=True)
    else:
        st.info("No open positions")
    
    # ========================
    # TRADE HISTORY
    # ========================
    st.markdown("---")
    st.markdown("## 📜 Trade History")
    
    if portfolio.trade_history:
        trade_df = pd.DataFrame([{
            'Symbol': t.stock,
            'Entry': f"${t.entry_price:.2f}",
            'Exit': f"${t.exit_price:.2f}",
            'Type': t.position_type,
            'P&L': f"${t.pnl:.2f}",
            'P&L %': f"{t.pnl_percent:.1f}%",
            'Reason': t.exit_reason
        } for t in portfolio.trade_history[-20:]])
        
        st.dataframe(trade_df, use_container_width=True)
        
        # Stats
        stats = portfolio.get_stats()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Trades", stats['total_trades'])
        c2.metric("Win Rate", f"{stats['win_rate']:.1f}%")
        c3.metric("Profit Factor", f"{stats['profit_factor']:.2f}")
        c4.metric("Total P&L", f"${stats['total_pnl']:.2f}")
        c5.metric("Current Capital", f"${stats['current_capital']:,.0f}")
    else:
        st.info("No trades yet")
    
    # ========================
    # AUTO REFRESH
    # ========================
    if auto_refresh:
        time.sleep(30)
        st.rerun()


if __name__ == "__main__":
    main()
