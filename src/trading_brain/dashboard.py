"""
AI Brain Pro - Professional Trading Terminal
============================================
A Bloomberg-style trading terminal with live data, advanced charting,
and AI-powered trading signals.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf
import time
from threading import Thread

st.set_page_config(
    page_title="AI Brain Pro | Trading Terminal",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - BLOOMBERG TERMINAL STYLE
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    :root {
        --bg-primary: #0a0e17;
        --bg-secondary: #0d1117;
        --bg-card: #161b22;
        --bg-card-hover: #1c2128;
        --accent-green: #00d68f;
        --accent-red: #ff3d71;
        --accent-blue: #0095ff;
        --accent-gold: #ffaa00;
        --text-primary: #ffffff;
        --text-secondary: #8b949e;
        --border-color: #30363d;
        --glow-green: rgba(0, 214, 143, 0.3);
        --glow-red: rgba(255, 61, 113, 0.3);
    }
    
    .stApp {
        background: var(--bg-primary);
    }
    
    /* Main Header */
    .terminal-header {
        background: linear-gradient(180deg, #0d1117 0%, #0a0e17 100%);
        border-bottom: 1px solid var(--border-color);
        padding: 12px 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        position: sticky;
        top: 0;
        z-index: 100;
    }
    
    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .brand h1 {
        font-family: 'JetBrains Mono', monospace;
        font-size: 20px;
        margin: 0;
        background: linear-gradient(90deg, #0095ff, #00d68f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .live-badge {
        background: var(--accent-green);
        color: #000;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    /* KPI Cards */
    .metric-card {
        background: linear-gradient(145deg, var(--bg-card) 0%, rgba(22, 27, 34, 0.7) 100%);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 16px 20px;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-green));
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .metric-card:hover::before {
        opacity: 1;
    }
    
    .metric-card:hover {
        border-color: var(--accent-blue);
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
    }
    
    .metric-label {
        font-size: 11px;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 24px;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    .metric-change {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        margin-top: 4px;
    }
    
    .positive { color: var(--accent-green); }
    .negative { color: var(--accent-red); }
    .neutral { color: var(--text-secondary); }
    
    /* Signal Badges */
    .signal-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 8px 16px;
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 14px;
        letter-spacing: 1px;
    }
    
    .signal-buy {
        background: linear-gradient(135deg, rgba(0, 214, 143, 0.2), rgba(0, 214, 143, 0.1));
        border: 2px solid var(--accent-green);
        color: var(--accent-green);
        box-shadow: 0 0 20px var(--glow-green);
    }
    
    .signal-sell {
        background: linear-gradient(135deg, rgba(255, 61, 113, 0.2), rgba(255, 61, 113, 0.1));
        border: 2px solid var(--accent-red);
        color: var(--accent-red);
        box-shadow: 0 0 20px var(--glow-red);
    }
    
    .signal-hold {
        background: linear-gradient(135deg, rgba(139, 148, 158, 0.2), rgba(139, 148, 158, 0.1));
        border: 2px solid var(--text-secondary);
        color: var(--text-secondary);
    }
    
    /* News Ticker */
    .news-ticker {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 16px;
    }
    
    .news-item {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 0;
        border-bottom: 1px solid var(--border-color);
    }
    
    .news-item:last-child {
        border-bottom: none;
    }
    
    .news-sentiment {
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: 600;
    }
    
    .sentiment-bullish { background: rgba(0, 214, 143, 0.2); color: var(--accent-green); }
    .sentiment-bearish { background: rgba(255, 61, 113, 0.2); color: var(--accent-red); }
    .sentiment-neutral { background: rgba(139, 148, 158, 0.2); color: var(--text-secondary); }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
        border-right: 1px solid var(--border-color);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        font-size: 14px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-card);
        border-radius: 8px;
        padding: 4px;
        gap: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 6px;
        padding: 10px 20px;
        font-weight: 500;
        color: var(--text-secondary);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.05);
        color: var(--text-primary);
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--accent-blue) !important;
        color: white !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-blue), #0066cc);
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 149, 255, 0.3);
    }
    
    .stButton > button[data-baseweb="button"] {
        border: none;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 3px;
    }
    
    /* Dataframes */
    [data-testid="stDataFrame"] {
        background: var(--bg-card);
        border-radius: 8px;
    }
    
    /* Gauge */
    .gauge-container {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    
    /* Portfolio card */
    .portfolio-card {
        background: linear-gradient(145deg, var(--bg-card) 0%, rgba(22, 27, 34, 0.8) 100%);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 20px;
    }
    
    /* Trade form */
    .trade-form {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 20px;
    }
    
    /* Expander styling */
    [data-testid="stExpander"] {
        background: var(--bg-card);
        border-radius: 8px;
        border: 1px solid var(--border-color);
    }
    
    /* Selectbox styling */
    [data-testid="stSelectbox"] [data-baseweb="select"] {
        background: var(--bg-card);
        border-color: var(--border-color);
    }
    
    /* Number input styling */
    [data-testid="stNumberInput"] [data-baseweb="input"] {
        background: var(--bg-card);
        border-color: var(--border-color);
    }
    
    /* Date input styling */
    [data-testid="stDateInput"] [data-baseweb="input"] {
        background: var(--bg-card);
        border-color: var(--border-color);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []

if 'open_positions' not in st.session_state:
    st.session_state.open_positions = []

if 'virtual_balance' not in st.session_state:
    st.session_state.virtual_balance = 100000.0

if 'market_data' not in st.session_state:
    st.session_state.market_data = {
        'sp500': {'price': 4890.23, 'change': 0.45},
        'nasdaq': {'price': 15432.56, 'change': 0.78},
        'dow': {'price': 38123.45, 'change': -0.23},
        'vix': {'price': 13.45, 'change': -2.1}
    }

if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()

if 'news_items' not in st.session_state:
    st.session_state.news_items = [
        ("Fed signals potential rate cut in Q2", "Bullish", "2h"),
        ("Tech stocks rally on strong earnings", "Bullish", "3h"),
        ("Oil prices stabilize amid tensions", "Neutral", "4h"),
        ("Crypto markets show recovery signs", "Bullish", "5h"),
        ("S&P 500 reaches new all-time high", "Bullish", "6h"),
    ]


# ============================================================================
# DATA FUNCTIONS
# ============================================================================

@st.cache_data(ttl=60)
def fetch_live_data(symbol: str, period: str = "2mo", interval: str = "1h") -> pd.DataFrame:
    """Fetch live stock data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators"""
    df = df.copy()
    
    # EMA calculations
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Volume SMA
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    
    return df


def generate_ai_signal(df: pd.DataFrame) -> tuple:
    """Generate AI trading signal based on RSI and MA crossover"""
    if len(df) < 50:
        return "HOLD", 50, {}
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # RSI Analysis
    rsi = latest['RSI']
    
    # EMA Crossover
    ema_cross = "above" if latest['EMA_20'] > latest['EMA_50'] else "below"
    
    # MACD Analysis
    macd_bullish = latest['MACD'] > latest['MACD_Signal']
    
    # Signal scoring
    score = 50
    
    # RSI contribution
    if rsi < 30:
        score += 20
    elif rsi > 70:
        score -= 20
    elif rsi < 45:
        score += 10
    elif rsi > 55:
        score -= 10
    
    # EMA contribution
    if ema_cross == "above":
        score += 15
    else:
        score -= 15
    
    # MACD contribution
    if macd_bullish:
        score += 15
    else:
        score -= 15
    
    # Determine signal
    if score >= 65 and rsi < 70:
        signal = "BUY"
    elif score <= 35 and rsi > 30:
        signal = "SELL"
    else:
        signal = "HOLD"
    
    details = {
        'rsi': rsi,
        'ema_cross': ema_cross,
        'macd_bullish': macd_bullish,
        'score': score,
        'trend': 'UPTREND' if latest['EMA_20'] > latest['EMA_50'] else 'DOWNTREND'
    }
    
    return signal, score, details


def calculate_portfolio_metrics():
    """Calculate portfolio metrics"""
    total_invested = 0
    current_value = 0
    unrealized_pnl = 0
    
    for pos in st.session_state.open_positions:
        total_invested += pos['entry_price'] * pos['quantity']
        current_value += pos['current_price'] * pos['quantity']
    
    unrealized_pnl = current_value - total_invested
    
    realized_pnl = sum(t['pnl'] for t in st.session_state.trade_history)
    
    return {
        'total_equity': st.session_state.virtual_balance + current_value,
        'realized_pnl': realized_pnl,
        'unrealized_pnl': unrealized_pnl,
        'cash': st.session_state.virtual_balance,
        'positions_value': current_value
    }


# ============================================================================
# CHART FUNCTIONS
# ============================================================================

def create_candlestick_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """Create professional candlestick chart with indicators"""
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=('', '', '', '')
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='#00d68f',
            decreasing_line_color='#ff3d71',
            increasing_fillcolor='#00d68f',
            decreasing_fillcolor='#ff3d71'
        ),
        row=1, col=1
    )
    
    # EMA 20
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['EMA_20'],
            name='EMA 20',
            line=dict(color='#0095ff', width=1.5)
        ),
        row=1, col=1
    )
    
    # EMA 50
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['EMA_50'],
            name='EMA 50',
            line=dict(color='#ffaa00', width=1.5)
        ),
        row=1, col=1
    )
    
    # Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['BB_Upper'],
            name='BB Upper',
            line=dict(color='#8b949e', width=1, dash='dash'),
            showlegend=False
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['BB_Lower'],
            name='BB Lower',
            line=dict(color='#8b949e', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(139, 148, 158, 0.08)',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['RSI'],
            name='RSI (14)',
            line=dict(color='#9966ff', width=2),
            fill='tozeroy',
            fillcolor='rgba(153, 102, 255, 0.1)'
        ),
        row=2, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="#ff3d71", line_width=1, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#00d68f", line_width=1, row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="#8b949e", line_width=1, row=2, col=1)
    
    # MACD
    colors = ['#00d68f' if val >= 0 else '#ff3d71' for val in df['MACD_Hist']]
    fig.add_trace(
        go.Bar(
            x=df.index, y=df['MACD_Hist'],
            name='MACD Hist',
            marker_color=colors,
            marker_opacity=0.7
        ),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['MACD'],
            name='MACD',
            line=dict(color='#0095ff', width=2)
        ),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['MACD_Signal'],
            name='Signal',
            line=dict(color='#ffaa00', width=2)
        ),
        row=3, col=1
    )
    
    # Volume
    vol_colors = ['#00d68f' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#ff3d71' 
                  for i in range(len(df))]
    fig.add_trace(
        go.Bar(
            x=df.index, y=df['Volume'],
            name='Volume',
            marker_color=vol_colors,
            marker_opacity=0.7
        ),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['Volume_SMA'],
            name='Vol SMA',
            line=dict(color='#8b949e', width=1, dash='dash'),
            showlegend=True
        ),
        row=4, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=700,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color='#8b949e')
        ),
        paper_bgcolor='#0a0e17',
        plot_bgcolor='#0a0e17',
        font=dict(family="Inter, sans-serif", size=11, color='#ffffff'),
        margin=dict(t=20, b=40, l=60, r=40)
    )
    
    # Update axes
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.03)', showticklabels=True)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.03)', showticklabels=True)
    
    return fig


def create_sentiment_gauge(bullish_score: int) -> go.Figure:
    """Create sentiment gauge chart"""
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=bullish_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#8b949e'},
            'bar': {'color': '#0095ff'},
            'bgcolor': '#161b22',
            'borderwidth': 2,
            'bordercolor': '#30363d',
            'steps': [
                {'range': [0, 33], 'color': 'rgba(255, 61, 113, 0.3)'},
                {'range': [33, 66], 'color': 'rgba(139, 148, 158, 0.3)'},
                {'range': [66, 100], 'color': 'rgba(0, 214, 143, 0.3)'}
            ],
            'threshold': {
                'line': {'color': '#ffffff', 'width': 4},
                'thickness': 0.8,
                'value': bullish_score
            }
        },
        number={'font': {'color': '#ffffff', 'size': 40}}
    ))
    
    fig.update_layout(
        height=200,
        paper_bgcolor='#0a0e17',
        plot_bgcolor='#0a0e17',
        font=dict(color='#ffffff'),
        showlegend=False,
        margin=dict(t=20, b=20, l=20, r=20)
    )
    
    return fig


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown("""
    <div class="terminal-header">
        <div class="brand">
            <h1>🧠 AI Brain Pro</h1>
            <span class="live-badge">● LIVE</span>
        </div>
        <div style="display: flex; align-items: center; gap: 20px;">
            <span style="font-family: 'JetBrains Mono'; color: var(--text-secondary); font-size: 12px;">
                {} EST
            </span>
            <span style="font-family: 'JetBrains Mono'; color: var(--accent-green); font-size: 12px;">
                ● Connected
            </span>
        </div>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Market Overview")
        
        # Market data cards
        mkt = st.session_state.market_data
        
        for name, data, key in [
            ("S&P 500", mkt['sp500'], "SPX"),
            ("NASDAQ", mkt['nasdaq'], "IXIC"),
            ("DOW JONES", mkt['dow'], "DJI"),
            ("VIX", mkt['vix'], "VIX")
        ]:
            change_class = "positive" if data['change'] >= 0 else "negative"
            change_sign = "+" if data['change'] >= 0 else ""
            st.markdown(f"""
            <div class="metric-card" style="padding: 12px 16px; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: var(--text-secondary); font-size: 11px;">{name}</span>
                    <span class="metric-change {change_class}" style="font-size: 10px;">
                        {change_sign}{data['change']:.2f}%
                    </span>
                </div>
                <div class="metric-value" style="font-size: 18px; margin-top: 4px;">
                    {data['price']:,.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # News Section
        st.markdown("### 📰 Market News")
        
        for headline, sentiment, time_ in st.session_state.news_items[:4]:
            sent_class = f"sentiment-{sentiment.lower()}"
            st.markdown(f"""
            <div class="news-item">
                <span class="news-sentiment {sent_class}">{sentiment}</span>
                <div style="flex: 1;">
                    <p style="margin: 0; font-size: 12px; color: var(--text-primary);">{headline}</p>
                    <span style="font-size: 10px; color: var(--text-secondary);">{time_} ago</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Sentiment Gauge
        st.markdown("### 🎯 Market Sentiment")
        sentiment_score = 55
        st.plotly_chart(create_sentiment_gauge(sentiment_score), use_container_width=True)
        
        st.markdown(f"""
        <div style="text-align: center; margin-top: -10px;">
            <span style="color: var(--accent-green); font-weight: 600;">Bullish</span>
            <span style="color: var(--text-secondary);"> | Neutral | </span>
            <span style="color: var(--accent-red);">Bearish</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Main Content Area
    st.markdown("---")
    
    # Symbol Selection
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        symbol_options = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM',
            'BTC-USD', 'ETH-USD', 'SOL-USD', 'SPY', 'QQQ', 'GLD', 'TQQQ'
        ]
        symbol = st.selectbox("Select Symbol", symbol_options, index=0)
    
    with col2:
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
    
    with col3:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
    
    # Portfolio Summary
    metrics = calculate_portfolio_metrics()
    
    col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)
    
    with col_p1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Equity</div>
            <div class="metric-value">${metrics['total_equity']:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_p2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Cash Balance</div>
            <div class="metric-value">${metrics['cash']:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_p3:
        unreal_class = "positive" if metrics['unrealized_pnl'] >= 0 else "negative"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Unrealized P&L</div>
            <div class="metric-value {unreal_class}">${metrics['unrealized_pnl']:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_p4:
        real_class = "positive" if metrics['realized_pnl'] >= 0 else "negative"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Realized P&L</div>
            <div class="metric-value {real_class}">${metrics['realized_pnl']:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_p5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Open Positions</div>
            <div class="metric-value">{len(st.session_state.open_positions)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Fetch and prepare data
    df = fetch_live_data(symbol, period=period)
    
    if not df.empty:
        df = calculate_indicators(df)
        signal, score, details = generate_ai_signal(df)
        latest_price = df['Close'].iloc[-1]
        price_change = df['Close'].iloc[-1] - df['Open'].iloc[0]
        price_change_pct = (price_change / df['Open'].iloc[0]) * 100
    else:
        df = pd.DataFrame()
        signal, score, details = "HOLD", 50, {}
        latest_price = 0
        price_change = 0
        price_change_pct = 0
    
    # Tabs for main content
    tab1, tab2, tab3 = st.tabs(["📈 Live Chart", "🔮 AI Analysis", "📜 Trade Logs"])
    
    with tab1:
        if not df.empty:
            # AI Signal Card
            col_s1, col_s2 = st.columns([3, 1])
            
            with col_s1:
                st.plotly_chart(create_candlestick_chart(df, symbol), use_container_width=True)
            
            with col_s2:
                st.markdown("### 🤖 AI Signal")
                
                signal_color = '#00d68f' if signal == 'BUY' else '#ff3d71' if signal == 'SELL' else '#8b949e'
                signal_bg = 'rgba(0, 214, 143, 0.2)' if signal == 'BUY' else 'rgba(255, 61, 113, 0.2)' if signal == 'SELL' else 'rgba(139, 148, 158, 0.2)'
                signal_border = '#00d68f' if signal == 'BUY' else '#ff3d71' if signal == 'SELL' else '#8b949e'
                
                st.html(f"""
                <div style="background: linear-gradient(135deg, {signal_bg}, rgba(22, 27, 34, 0.7)); border: 2px solid {signal_border}; border-radius: 8px; padding: 12px 20px; text-align: center; margin-bottom: 16px;">
                    <span style="font-family: 'JetBrains Mono'; font-size: 18px; font-weight: 700; color: {signal_color}; letter-spacing: 2px;">
                        {signal}
                    </span>
                </div>
                <div style="background: linear-gradient(145deg, #161b22 0%, rgba(22, 27, 34, 0.7) 100%); border: 1px solid #30363d; border-radius: 12px; padding: 16px; margin-top: 16px;">
                    <div style="color: #8b949e; font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 8px;">Signal Score</div>
                    <div style="font-family: 'JetBrains Mono'; font-size: 28px; font-weight: 700; color: #fff;">{score}/100</div>
                    <div style="color: #8b949e; font-size: 12px; margin-top: 4px;">Confidence: {abs(score-50)*2}%</div>
                </div>
                """)
                
                st.markdown("---")
                
                if details:
                    st.html(f"""
                    <div style="font-size: 13px; color: #fff;">
                        <p><strong>RSI (14):</strong> <span style="color: {'#00d68f' if details['rsi'] < 50 else '#ff3d71'};"> {details['rsi']:.1f}</span></p>
                        <p><strong>Trend:</strong> <span style="color: {'#00d68f' if details['trend'] == 'UPTREND' else '#ff3d71'};">{details['trend']}</span></p>
                        <p><strong>EMA:</strong> <span style="color: {'#00d68f' if details['ema_cross'] == 'above' else '#ff3d71'};">{details['ema_cross'].upper()} EMA</span></p>
                        <p><strong>MACD:</strong> <span style="color: {'#00d68f' if details['macd_bullish'] else '#ff3d71'};">{"BULLISH" if details['macd_bullish'] else "BEARISH"}</span></p>
                    </div>
                    """)
        else:
            st.warning("No data available for the selected symbol. Please try another.")
    
    with tab2:
        col_ai1, col_ai2 = st.columns([1, 1])
        
        with col_ai1:
            st.markdown("### 🧠 AI Trading Strategy")
            
            if details:
                price_change_class = "positive" if price_change >= 0 else "negative"
                rsi_class = "positive" if details['rsi'] < 50 else "negative"
                macd_class = "positive" if details['macd_bullish'] else "negative"
                macd_sign = "+" if df['MACD'].iloc[-1] >= 0 else ""
                
                st.html(f"""
                <div style="background: linear-gradient(145deg, #161b22 0%, rgba(22, 27, 34, 0.8) 100%); border: 1px solid #30363d; border-radius: 12px; padding: 20px;">
                    <h4 style="margin-top: 0; color: #fff;">Signal Analysis for {symbol}</h4>
                    <hr style="border-color: #30363d;">
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px;">
                        <div>
                            <p style="color: #8b949e; margin: 0; font-size: 11px;">CURRENT PRICE</p>
                            <p style="font-family: 'JetBrains Mono'; font-size: 24px; margin: 4px 0; color: #fff;">${latest_price:.2f}</p>
                            <p style="color: {'#00d68f' if price_change >= 0 else '#ff3d71'}; margin: 0; font-size: 12px;">
                                {price_change:+.2f} ({price_change_pct:+.2f}%)
                            </p>
                        </div>
                        <div>
                            <p style="color: #8b949e; margin: 0; font-size: 11px;">RECOMMENDATION</p>
                            <p style="font-size: 20px; margin: 4px 0; color: {'#00d68f' if signal == 'BUY' else '#ff3d71' if signal == 'SELL' else '#8b949e'};">{signal}</p>
                            <p style="margin: 0; font-size: 12px; color: #8b949e;">Score: {score}/100</p>
                        </div>
                    </div>
                    
                    <hr style="border-color: #30363d; margin: 20px 0;">
                    
                    <h5 style="margin-bottom: 12px; color: #fff;">Technical Indicators</h5>
                    <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                        <span style="background: #161b22; padding: 6px 12px; border-radius: 6px; font-size: 12px; color: #fff;">
                            RSI: <strong style="color: {'#00d68f' if details['rsi'] < 50 else '#ff3d71'};">{details['rsi']:.1f}</strong>
                        </span>
                        <span style="background: #161b22; padding: 6px 12px; border-radius: 6px; font-size: 12px; color: #fff;">
                            EMA 20: <strong>${df['EMA_20'].iloc[-1]:.2f}</strong>
                        </span>
                        <span style="background: #161b22; padding: 6px 12px; border-radius: 6px; font-size: 12px; color: #fff;">
                            EMA 50: <strong>${df['EMA_50'].iloc[-1]:.2f}</strong>
                        </span>
                        <span style="background: #161b22; padding: 6px 12px; border-radius: 6px; font-size: 12px; color: #fff;">
                            MACD: <strong style="color: {'#00d68f' if details['macd_bullish'] else '#ff3d71'};">{macd_sign}{df['MACD'].iloc[-1]:.4f}</strong>
                        </span>
                        <span style="background: #161b22; padding: 6px 12px; border-radius: 6px; font-size: 12px; color: #fff;">
                            BB Upper: <strong>${df['BB_Upper'].iloc[-1]:.2f}</strong>
                        </span>
                        <span style="background: #161b22; padding: 6px 12px; border-radius: 6px; font-size: 12px; color: #fff;">
                            BB Lower: <strong>${df['BB_Lower'].iloc[-1]:.2f}</strong>
                        </span>
                    </div>
                </div>
                """)
        
        with col_ai2:
            st.markdown("### 📊 Prediction Engine")
            
            st.info("🔮 AI price prediction coming soon with XGBoost ML model integration!")
            
            st.markdown("""
            <div class="portfolio-card" style="margin-top: 16px;">
                <h5 style="margin-bottom: 12px;">Strategy Logic</h5>
                <ol style="color: var(--text-secondary); font-size: 13px; line-height: 1.8;">
                    <li>RSI < 30 indicates <span style="color: var(--accent-green);">oversold</span> → potential BUY</li>
                    <li>RSI > 70 indicates <span style="color: var(--accent-red);">overbought</span> → potential SELL</li>
                    <li>EMA 20 crossing above EMA 50 → <span style="color: var(--accent-green);">Bullish</span> signal</li>
                    <li>EMA 20 crossing below EMA 50 → <span style="color: var(--accent-red);">Bearish</span> signal</li>
                    <li>MACD histogram positive → <span style="color: var(--accent-green);">Momentum building</span></li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        # Trade Execution Form
        col_t1, col_t2 = st.columns([1, 2])
        
        with col_t1:
            st.markdown("### 💹 Execute Trade")
            
            with st.form("trade_form"):
                trade_type = st.radio("Trade Type", ["BUY", "SELL"], horizontal=True)
                quantity = st.number_input("Quantity", min_value=1, value=10, step=1, key="qty")
                order_type = st.selectbox("Order Type", ["MARKET", "LIMIT"])
                
                if order_type == "LIMIT":
                    limit_price = st.number_input("Limit Price", value=float(latest_price), format="%.2f")
                else:
                    limit_price = latest_price
                
                if st.form_submit_button("📊 Execute Trade", use_container_width=True):
                    total_cost = limit_price * quantity
                    
                    if trade_type == "BUY":
                        if total_cost <= st.session_state.virtual_balance:
                            st.session_state.virtual_balance -= total_cost
                            st.session_state.open_positions.append({
                                'symbol': symbol,
                                'quantity': quantity,
                                'entry_price': limit_price,
                                'current_price': limit_price,
                                'type': 'LONG',
                                'timestamp': datetime.now()
                            })
                            st.success(f"✅ BUY order executed: {quantity} {symbol} @ ${limit_price:.2f}")
                        else:
                            st.error("❌ Insufficient funds!")
                    else:
                        # Sell from existing position or short
                        if any(p['symbol'] == symbol for p in st.session_state.open_positions):
                            for pos in st.session_state.open_positions:
                                if pos['symbol'] == symbol and pos['quantity'] >= quantity:
                                    pnl = (limit_price - pos['entry_price']) * quantity
                                    st.session_state.virtual_balance += (pos['entry_price'] * quantity) + pnl
                                    st.session_state.trade_history.append({
                                        'symbol': symbol,
                                        'type': 'SELL',
                                        'quantity': quantity,
                                        'entry': pos['entry_price'],
                                        'exit': limit_price,
                                        'pnl': pnl,
                                        'timestamp': datetime.now()
                                    })
                                    pos['quantity'] -= quantity
                                    if pos['quantity'] <= 0:
                                        st.session_state.open_positions.remove(pos)
                                    st.success(f"✅ SELL order executed: {quantity} {symbol} @ ${limit_price:.2f}")
                                    break
                        else:
                            # Short selling
                            st.session_state.open_positions.append({
                                'symbol': symbol,
                                'quantity': -quantity,
                                'entry_price': limit_price,
                                'current_price': limit_price,
                                'type': 'SHORT',
                                'timestamp': datetime.now()
                            })
                            st.session_state.trade_history.append({
                                'symbol': symbol,
                                'type': 'SHORT SELL',
                                'quantity': quantity,
                                'entry': limit_price,
                                'exit': None,
                                'pnl': 0,
                                'timestamp': datetime.now()
                            })
                            st.success(f"✅ SHORT order executed: {quantity} {symbol} @ ${limit_price:.2f}")
        
        with col_t2:
            # Open Positions
            st.markdown("### 📁 Open Positions")
            
            if st.session_state.open_positions:
                positions_df = pd.DataFrame(st.session_state.open_positions)
                positions_df['current_value'] = positions_df['current_price'] * positions_df['quantity']
                positions_df['unrealized_pnl'] = (positions_df['current_price'] - positions_df['entry_price']) * positions_df['quantity']
                positions_df['pnl_pct'] = ((positions_df['current_price'] - positions_df['entry_price']) / positions_df['entry_price'] * 100)
                
                # Format for display
                display_df = pd.DataFrame({
                    'Symbol': positions_df['symbol'],
                    'Qty': positions_df['quantity'],
                    'Entry': positions_df['entry_price'].apply(lambda x: f"${x:.2f}"),
                    'Current': positions_df['current_price'].apply(lambda x: f"${x:.2f}"),
                    'P&L': positions_df['unrealized_pnl'].apply(lambda x: f"${x:.2f}"),
                    'P&L %': positions_df['pnl_pct'].apply(lambda x: f"{x:.2f}%")
                })
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info("No open positions. Execute a trade to get started!")
            
            st.markdown("---")
            
            # Trade History
            st.markdown("### 📜 Trade History")
            
            if st.session_state.trade_history:
                history_df = pd.DataFrame(st.session_state.trade_history)
                
                display_hist = pd.DataFrame({
                    'Time': pd.to_datetime(history_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M'),
                    'Symbol': history_df['symbol'],
                    'Type': history_df['type'],
                    'Qty': history_df['quantity'],
                    'Entry': history_df['entry'].apply(lambda x: f"${x:.2f}" if x else '-'),
                    'Exit': history_df['exit'].apply(lambda x: f"${x:.2f}" if x else '-'),
                    'P&L': history_df['pnl'].apply(lambda x: f"${x:.2f}")
                })
                
                st.dataframe(display_hist, use_container_width=True, hide_index=True)
            else:
                st.info("No trade history yet.")


if __name__ == "__main__":
    main()
