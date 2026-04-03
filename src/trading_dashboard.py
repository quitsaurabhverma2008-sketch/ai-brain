"""
Trading AI Brain - Live Dashboard
==================================
Real-time trading signals, portfolio tracking, and performance analytics

Run: streamlit run trading_dashboard.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="Trading AI Brain",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #0e1117
    }
    .stMetric {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #3d3d5c;
    }
    .positive {
        color: #00cc66;
    }
    .negative {
        color: #ff4b4b;
    }
    .sidebar-content {
        background-color: #1e1e2f;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA FUNCTIONS
# =============================================================================

@st.cache_data(ttl=60)  # Cache for 60 seconds
def get_stock_data(symbol, period="1y"):
    """Get stock data from yfinance"""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        return df
    except:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_multiple_stocks(symbols):
    """Get data for multiple stocks"""
    data = {}
    for symbol in symbols:
        df = get_stock_data(symbol)
        if not df.empty:
            data[symbol] = df
    return data


def calculate_indicators(df):
    """Calculate technical indicators"""
    df = df.copy()
    
    # Moving averages
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['BB_Mid'] = df['Close'].rolling(20).mean()
    df['BB_Std'] = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)
    
    return df


def generate_signals(df):
    """Generate trading signals"""
    if df.empty or len(df) < 50:
        return None, 0
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    buy_score = 0
    sell_score = 0
    
    trend = 'UP' if latest['SMA_20'] > latest['SMA_50'] else 'DOWN'
    
    # RSI signals
    if latest['RSI'] < 30:
        buy_score += 2
    if latest['RSI'] > 70:
        sell_score += 2
    
    # MACD signals
    if latest['MACD'] > latest['MACD_Signal']:
        buy_score += 1
    if latest['MACD'] < latest['MACD_Signal']:
        sell_score += 1
    if prev['MACD'] <= prev['MACD_Signal'] and latest['MACD'] > latest['MACD_Signal']:
        buy_score += 2
    if prev['MACD'] >= prev['MACD_Signal'] and latest['MACD'] < latest['MACD_Signal']:
        sell_score += 2
    
    # Price signals
    if latest['Close'] > latest['SMA_20']:
        buy_score += 1
    if latest['Close'] < latest['SMA_20']:
        sell_score += 1
    
    # Generate signal
    signal = None
    score = 0
    if buy_score >= 4 and trend == 'UP':
        signal = 'BUY'
        score = buy_score
    elif sell_score >= 4 and trend == 'DOWN':
        signal = 'SELL'
        score = sell_score
    
    return signal, score


# =============================================================================
# CHART FUNCTIONS
# =============================================================================

def create_candlestick_chart(df, symbol):
    """Create interactive candlestick chart"""
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))
    
    # Moving averages
    if 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA_20'],
            mode='lines', name='SMA 20',
            line=dict(color='#00cc66', width=1)
        ))
    
    if 'SMA_50' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA_50'],
            mode='lines', name='SMA 50',
            line=dict(color='#ff6b6b', width=1)
        ))
    
    # Bollinger Bands
    if 'BB_Upper' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_Upper'],
            mode='lines', name='BB Upper',
            line=dict(color='#666666', width=1),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_Lower'],
            mode='lines', name='BB Bands',
            line=dict(color='#666666', width=1),
            fill='tonexty',
            fillcolor='rgba(255,255,255,0.05)'
        ))
    
    fig.update_layout(
        title=f'{symbol} - Price Chart',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template='plotly_dark',
        height=500,
        xaxis_rangeslider_visible=False
    )
    
    return fig


def create_rsi_chart(df):
    """Create RSI chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['RSI'],
        mode='lines', name='RSI',
        line=dict(color='#ffa500', width=2)
    ))
    
    # Overbought line
    fig.add_hline(y=70, line_dash="dash", line_color="#ff4b4b", annotation_text="Overbought")
    # Oversold line
    fig.add_hline(y=30, line_dash="dash", line_color="#00cc66", annotation_text="Oversold")
    
    fig.update_layout(
        title='RSI (14)',
        xaxis_title='Date',
        yaxis_title='RSI',
        template='plotly_dark',
        height=200,
        yaxis=dict(range=[0, 100])
    )
    
    return fig


def create_macd_chart(df):
    """Create MACD chart"""
    fig = go.Figure()
    
    # MACD line
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MACD'],
        mode='lines', name='MACD',
        line=dict(color='#00ccff', width=2)
    ))
    
    # Signal line
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MACD_Signal'],
        mode='lines', name='Signal',
        line=dict(color='#ffa500', width=2)
    ))
    
    # Histogram
    colors = ['#00cc66' if val >= 0 else '#ff4b4b' for val in df['MACD_Hist']]
    fig.add_trace(go.Bar(
        x=df.index, y=df['MACD_Hist'],
        name='Histogram',
        marker_color=colors
    ))
    
    fig.update_layout(
        title='MACD',
        xaxis_title='Date',
        yaxis_title='MACD',
        template='plotly_dark',
        height=250
    )
    
    return fig


def create_portfolio_chart(portfolio_value):
    """Create portfolio performance chart"""
    df = pd.DataFrame(portfolio_value, columns=['value'])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df['value'],
        mode='lines',
        fill='tozeroy',
        line=dict(color='#00cc66', width=2),
        fillcolor='rgba(0,204,102,0.2)'
    ))
    
    fig.update_layout(
        title='Portfolio Value',
        xaxis_title='Trade',
        yaxis_title='Value ($)',
        template='plotly_dark',
        height=300
    )
    
    return fig


# =============================================================================
# SIDEBAR
# =============================================================================

def create_sidebar():
    """Create sidebar with settings"""
    with st.sidebar:
        st.title("📈 Trading AI Brain")
        st.markdown("---")
        
        st.subheader("⚙️ Settings")
        
        # Stock selection
        stocks = st.multiselect(
            "Select Stocks",
            ['GOOGL', 'META', 'AAPL', 'MSFT', 'SPY', 'AMZN', 'NVDA', 'TSLA', 'NFLX', 'AMD', 'JPM', 'V', 'MA', 'PYPL'],
            default=['GOOGL', 'META', 'AAPL', 'MSFT']
        )
        
        # Time period
        period = st.selectbox(
            "Time Period",
            ['1mo', '3mo', '6mo', '1y', '2y', '5y'],
            index=3
        )
        
        st.markdown("---")
        
        st.subheader("💰 Portfolio")
        initial_capital = st.number_input("Initial Capital ($)", value=100000, step=10000)
        
        st.markdown("---")
        
        st.subheader("📊 Backtest Results")
        if st.button("Load Results"):
            try:
                df = pd.read_csv("phase1_walkforward_results.csv")
                st.dataframe(df, height=200)
            except:
                st.warning("No results file found")
        
        st.markdown("---")
        st.caption("Data refreshes every 60 seconds")
        
        return stocks, period, initial_capital


# =============================================================================
# MAIN DASHBOARD
# =============================================================================

def main():
    # Get settings from sidebar
    stocks, period, initial_capital = create_sidebar()
    
    # Header
    st.title("📈 Trading AI Brain - Live Dashboard")
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not stocks:
        st.warning("Please select stocks from sidebar")
        return
    
    # Get data for all stocks
    stock_data = get_multiple_stocks(stocks)
    
    if not stock_data:
        st.error("No data available. Please check your internet connection.")
        return
    
    # ========================
    # METRICS ROW
    # ========================
    st.markdown("### 📊 Market Overview")
    
    cols = st.columns(len(stocks))
    
    for idx, symbol in enumerate(stocks):
        if symbol in stock_data:
            df = stock_data[symbol]
            if not df.empty:
                latest = df.iloc[-1]
                prev = df.iloc[-10] if len(df) > 10 else df.iloc[0]
                
                price = latest['Close']
                change = ((price - prev['Close']) / prev['Close']) * 100
                volume = latest['Volume']
                
                with cols[idx]:
                    st.metric(
                        label=symbol,
                        value=f"${price:.2f}",
                        delta=f"{change:.2f}%"
                    )
    
    st.markdown("---")
    
    # ========================
    # SIGNAL SUMMARY
    # ========================
    st.markdown("### 🎯 Trading Signals")
    
    signal_data = []
    
    for symbol in stocks:
        if symbol in stock_data:
            df = stock_data[symbol]
            if not df.empty and len(df) > 50:
                df = calculate_indicators(df)
                signal, score = generate_signals(df)
                
                latest = df.iloc[-1]
                
                signal_data.append({
                    'Symbol': symbol,
                    'Price': latest['Close'],
                    'RSI': latest['RSI'],
                    'MACD': 'Above' if latest['MACD'] > latest['MACD_Signal'] else 'Below',
                    'Trend': 'UP' if latest['SMA_20'] > latest['SMA_50'] else 'DOWN',
                    'Signal': signal if signal else 'HOLD',
                    'Score': score if score else 0
                })
    
    if signal_data:
        signal_df = pd.DataFrame(signal_data)
        
        # Color code signals using pandas map
        def color_signal(val):
            if val == 'BUY':
                return 'background-color: #00cc66; color: white'
            elif val == 'SELL':
                return 'background-color: #ff4b4b; color: white'
            return ''
        
        try:
            # Try new pandas map method
            st.dataframe(
                signal_df.style.map(color_signal, subset=['Signal']),
                use_container_width=True
            )
        except:
            # Fallback to simple display
            st.dataframe(signal_df, use_container_width=True)
    
    st.markdown("---")
    
    # ========================
    # CHARTS
    # ========================
    
    # Select stock for detailed chart
    selected_stock = st.selectbox("Select stock for detailed chart", stocks)
    
    if selected_stock in stock_data:
        df = stock_data[selected_stock]
        
        if not df.empty and len(df) > 50:
            df = calculate_indicators(df)
            
            # Main chart
            fig = create_candlestick_chart(df, selected_stock)
            st.plotly_chart(fig, use_container_width=True)
            
            # Indicator charts
            col1, col2 = st.columns(2)
            
            with col1:
                rsi_fig = create_rsi_chart(df)
                st.plotly_chart(rsi_fig, use_container_width=True)
            
            with col2:
                macd_fig = create_macd_chart(df)
                st.plotly_chart(macd_fig, use_container_width=True)
    
    # ========================
    # PORTFOLIO SIMULATOR
    # ========================
    st.markdown("---")
    st.markdown("### 💼 Portfolio Simulator")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Initial Capital", f"${initial_capital:,.0f}")
    
    with col2:
        # Calculate mock portfolio
        total_value = initial_capital
        for sig in signal_data:
            if sig['Signal'] == 'BUY':
                total_value *= 1.02  # Assume 2% gain per buy signal
            elif sig['Signal'] == 'SELL':
                total_value *= 0.98  # Assume 2% loss per sell signal
        
        pnl = total_value - initial_capital
        pnl_pct = (pnl / initial_capital) * 100
        
        st.metric("Current Value", f"${total_value:,.0f}", f"{pnl_pct:.2f}%")
    
    with col3:
        buy_signals = sum(1 for s in signal_data if s['Signal'] == 'BUY')
        st.metric("Buy Signals", buy_signals)
    
    with col4:
        sell_signals = sum(1 for s in signal_data if s['Signal'] == 'SELL')
        st.metric("Sell Signals", sell_signals)
    
    # ========================
    # STOCK COMPARISON
    # ========================
    st.markdown("---")
    st.markdown("### 📈 Stock Comparison")
    
    comparison_data = []
    
    for symbol in stocks:
        if symbol in stock_data:
            df = stock_data[symbol]
            if not df.empty:
                latest = df.iloc[-1]
                week_change = ((latest['Close'] - df.iloc[-5]['Close']) / df.iloc[-5]['Close']) * 100 if len(df) > 5 else 0
                month_change = ((latest['Close'] - df.iloc[-20]['Close']) / df.iloc[-20]['Close']) * 100 if len(df) > 20 else 0
                
                comparison_data.append({
                    'Symbol': symbol,
                    'Price': latest['Close'],
                    'Week %': week_change,
                    'Month %': month_change,
                    'Volume': latest['Volume']
                })
    
    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        
        # Create comparison chart
        fig = go.Figure()
        
        for _, row in comp_df.iterrows():
            fig.add_trace(go.Bar(
                x=[row['Symbol']],
                y=[row['Week %']],
                name=row['Symbol'],
                text=f"{row['Week %']:.2f}%",
                textposition='auto'
            ))
        
        fig.update_layout(
            title='Weekly Performance (%)',
            template='plotly_dark',
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.dataframe(comp_df, use_container_width=True)
    
    # ========================
    # FOOTER
    # ========================
    st.markdown("---")
    st.markdown("""
    ---
    **Trading AI Brain** - Phase 1: Robust Backtesting
    
    Data provided by Yahoo Finance. For educational purposes only.
    """)


if __name__ == "__main__":
    main()
