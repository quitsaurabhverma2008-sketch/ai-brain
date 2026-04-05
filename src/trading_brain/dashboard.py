"""
AI Brain Pro - Professional Trading Terminal
============================================
A Bloomberg-style fintech application with live data, advanced charting,
AI-powered trading signals, and paper trading simulation.
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
import io

st.set_page_config(
    page_title="AI Brain Pro | Trading Terminal",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - BLOOMBERG TERMINAL STYLE + GLASSMORPHISM
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
    }
    
    .stApp {
        background: var(--bg-primary);
    }
    
    /* Header */
    .terminal-header {
        background: linear-gradient(180deg, #0d1117 0%, #0a0e17 100%);
        border-bottom: 1px solid var(--border-color);
        padding: 12px 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: -1rem -1rem 1rem -1rem;
    }
    
    .brand h1 {
        font-family: 'JetBrains Mono', monospace;
        font-size: 22px;
        margin: 0;
        background: linear-gradient(90deg, #0095ff, #00d68f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(22, 27, 34, 0.8);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(0, 149, 255, 0.5);
        box-shadow: 0 12px 40px rgba(0, 149, 255, 0.15);
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, rgba(22, 27, 34, 0.9) 0%, rgba(16, 22, 28, 0.7) 100%);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 16px 20px;
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-green));
    }
    
    .metric-label {
        font-size: 10px;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 22px;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.2;
    }
    
    .metric-change {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        margin-top: 4px;
    }
    
    .positive { color: var(--accent-green); }
    .negative { color: var(--accent-red); }
    .neutral { color: var(--text-secondary); }
    
    /* Live Indicator */
    .live-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: var(--accent-green);
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(0, 214, 143, 0.7); }
        50% { opacity: 0.8; box-shadow: 0 0 0 8px rgba(0, 214, 143, 0); }
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #0a0e17 100%);
        border-right: 1px solid var(--border-color);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-card);
        border-radius: 10px;
        padding: 5px;
        gap: 5px;
        border: 1px solid var(--border-color);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        color: var(--text-secondary);
        transition: all 0.2s;
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
        background: linear-gradient(135deg, var(--accent-blue), #0070cc);
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 149, 255, 0.3);
    }
    
    /* News Badge */
    .news-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-bullish {
        background: rgba(0, 214, 143, 0.2);
        color: var(--accent-green);
    }
    
    .badge-bearish {
        background: rgba(255, 61, 113, 0.2);
        color: var(--accent-red);
    }
    
    .badge-neutral {
        background: rgba(139, 148, 158, 0.2);
        color: var(--text-secondary);
    }
    
    /* News Item */
    .news-item {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
        transition: all 0.2s;
    }
    
    .news-item:hover {
        border-color: var(--accent-blue);
        background: var(--bg-card-hover);
    }
    
    /* Selectbox & Inputs */
    [data-testid="stSelectbox"] [data-baseweb="select"] {
        background: var(--bg-card);
        border-color: var(--border-color);
    }
    
    /* DataFrame */
    [data-testid="stDataFrame"] {
        background: var(--bg-card);
        border-radius: 8px;
    }
    
    /* Custom Scrollbar */
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
    
    /* Signal Badge */
    .signal-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 12px 24px;
        border-radius: 10px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 16px;
        letter-spacing: 2px;
        min-width: 120px;
    }
    
    .signal-buy {
        background: linear-gradient(135deg, rgba(0, 214, 143, 0.2), rgba(0, 214, 143, 0.1));
        border: 2px solid var(--accent-green);
        color: var(--accent-green);
        box-shadow: 0 0 30px rgba(0, 214, 143, 0.3);
    }
    
    .signal-sell {
        background: linear-gradient(135deg, rgba(255, 61, 113, 0.2), rgba(255, 61, 113, 0.1));
        border: 2px solid var(--accent-red);
        color: var(--accent-red);
        box-shadow: 0 0 30px rgba(255, 61, 113, 0.3);
    }
    
    .signal-hold {
        background: linear-gradient(135deg, rgba(139, 148, 158, 0.2), rgba(139, 148, 158, 0.1));
        border: 2px solid var(--text-secondary);
        color: var(--text-secondary);
    }
    
    /* Toggle Switch */
    .toggle-switch {
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 4px 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'virtual_balance' not in st.session_state:
    st.session_state.virtual_balance = 100000.0

if 'open_positions' not in st.session_state:
    st.session_state.open_positions = []

if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []

if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()


# ============================================================================
# DATA FUNCTIONS
# ============================================================================

@st.cache_data(ttl=60)
def fetch_live_data(symbol: str, period: str = "3mo", interval: str = "1h") -> pd.DataFrame:
    """Fetch live stock data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators"""
    df = df.copy()
    
    # EMA
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
    
    return df


def generate_ai_signal(df: pd.DataFrame) -> tuple:
    """Generate AI trading signal"""
    if len(df) < 50:
        return "HOLD", 50, {}
    
    latest = df.iloc[-1]
    score = 50
    
    # RSI Analysis (30 points max)
    rsi = latest['RSI']
    if rsi < 30:
        score += 25
    elif rsi < 40:
        score += 15
    elif rsi > 70:
        score -= 25
    elif rsi > 60:
        score -= 15
    
    # EMA Crossover (30 points max)
    ema_cross = "above" if latest['EMA_20'] > latest['EMA_50'] else "below"
    if ema_cross == "above":
        score += 20
    else:
        score -= 20
    
    # MACD Analysis (20 points max)
    macd_bullish = latest['MACD'] > latest['MACD_Signal']
    if macd_bullish:
        score += 15
    else:
        score -= 15
    
    # MACD Histogram direction
    if len(df) > 1:
        prev_hist = df['MACD_Hist'].iloc[-2]
        curr_hist = latest['MACD_Hist']
        if curr_hist > prev_hist > 0 or curr_hist > prev_hist < 0:
            score += 5
        elif curr_hist < prev_hist < 0 or curr_hist < prev_hist > 0:
            score -= 5
    
    # Determine signal
    if score >= 65:
        signal = "BUY"
    elif score <= 35:
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


def analyze_sentiment(headline: str) -> str:
    """Simple keyword-based sentiment analysis"""
    bullish_words = ['rally', 'surge', 'gain', 'bullish', 'high', 'rise', 'growth', 'profit', 'beat', 'upgrade', 'strong', 'positive', 'recovery', 'all-time']
    bearish_words = ['fall', 'drop', 'loss', 'bearish', 'low', 'decline', 'weak', 'miss', 'downgrade', 'negative', 'concern', 'tension', 'fear']
    
    headline_lower = headline.lower()
    
    bullish_count = sum(1 for word in bullish_words if word in headline_lower)
    bearish_count = sum(1 for word in bearish_words if word in headline_lower)
    
    if bullish_count > bearish_count:
        return "Bullish"
    elif bearish_count > bullish_count:
        return "Bearish"
    return "Neutral"


def get_market_news() -> list:
    """Get market news with sentiment"""
    news = [
        ("Fed signals potential rate cut in Q2", "Bullish"),
        ("Tech stocks rally on strong earnings", "Bullish"),
        ("Oil prices stabilize amid tensions", "Neutral"),
        ("Crypto markets show recovery signs", "Bullish"),
        ("S&P 500 reaches new all-time high", "Bullish"),
        ("Treasury yields fall on weak data", "Bullish"),
        ("Retail sales exceed expectations", "Bullish"),
        ("Manufacturing PMI shows expansion", "Bullish"),
    ]
    return [(h, analyze_sentiment(h)) for h, _ in news]


def calculate_portfolio_metrics(df_current: pd.DataFrame = None) -> dict:
    """Calculate portfolio metrics"""
    unrealized_pnl = 0
    positions_value = 0
    
    for pos in st.session_state.open_positions:
        if df_current is not None and len(df_current) > 0:
            try:
                live_data = fetch_live_data(pos['symbol'], period="1d", interval="1m")
                if not live_data.empty:
                    current_price = live_data['Close'].iloc[-1]
                else:
                    current_price = pos['current_price']
            except:
                current_price = pos['current_price']
        else:
            current_price = pos['current_price']
        
        pos['current_price'] = current_price
        pnl = (current_price - pos['entry_price']) * pos['quantity']
        unrealized_pnl += pnl
        positions_value += current_price * pos['quantity']
    
    realized_pnl = sum(t['pnl'] for t in st.session_state.trade_history)
    
    return {
        'total_equity': st.session_state.virtual_balance + positions_value + unrealized_pnl,
        'cash': st.session_state.virtual_balance,
        'unrealized_pnl': unrealized_pnl,
        'realized_pnl': realized_pnl,
        'positions_value': positions_value
    }


# ============================================================================
# CHART FUNCTIONS
# ============================================================================

def create_main_chart(df: pd.DataFrame, show_ema20: bool = True, show_ema50: bool = True, 
                      show_bb: bool = True) -> go.Figure:
    """Create professional candlestick chart with indicators"""
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=('', '', '')
    )
    
    # Candlestick
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
            increasing_fillcolor='rgba(0, 214, 143, 0.3)',
            decreasing_fillcolor='rgba(255, 61, 113, 0.3)'
        ),
        row=1, col=1
    )
    
    # EMA 20
    if show_ema20 and 'EMA_20' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['EMA_20'], name='EMA 20',
                      line=dict(color='#0095ff', width=1.5)),
            row=1, col=1
        )
    
    # EMA 50
    if show_ema50 and 'EMA_50' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['EMA_50'], name='EMA 50',
                      line=dict(color='#ffaa00', width=1.5)),
            row=1, col=1
        )
    
    # Bollinger Bands
    if show_bb and 'BB_Upper' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper',
                      line=dict(color='#8b949e', width=1, dash='dash'),
                      showlegend=False),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower',
                      line=dict(color='#8b949e', width=1, dash='dash'),
                      fill='tonexty', fillcolor='rgba(139, 148, 158, 0.05)',
                      showlegend=False),
            row=1, col=1
        )
    
    # RSI
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                      line=dict(color='#9966ff', width=2),
                      fill='tozeroy', fillcolor='rgba(153, 102, 255, 0.1)'),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="#ff3d71", line_width=1, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#00d68f", line_width=1, row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="#8b949e", line_width=1, row=2, col=1)
    
    # MACD
    if 'MACD' in df.columns:
        colors = ['#00d68f' if val >= 0 else '#ff3d71' for val in df['MACD_Hist']]
        fig.add_trace(
            go.Bar(x=df.index, y=df['MACD_Hist'], name='MACD Hist',
                  marker_color=colors, marker_opacity=0.7),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                      line=dict(color='#0095ff', width=2)),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal',
                      line=dict(color='#ffaa00', width=2)),
            row=3, col=1
        )
    
    # Layout
    fig.update_layout(
        height=550,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                   bgcolor="rgba(0,0,0,0)", font=dict(color='#8b949e')),
        paper_bgcolor='#0a0e17',
        plot_bgcolor='#0a0e17',
        font=dict(family="Inter, sans-serif", size=11, color='#ffffff'),
        margin=dict(t=10, b=40, l=60, r=40)
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.03)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.03)')
    
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
        </div>
        <div style="display: flex; align-items: center; gap: 20px;">
            <span style="font-family: 'JetBrains Mono'; color: var(--accent-green); font-size: 12px;">
                <span class="live-dot"></span>LIVE
            </span>
            <span style="font-family: 'JetBrains Mono'; color: var(--text-secondary); font-size: 11px;">
                {} EST
            </span>
        </div>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Market Overview")
        
        # Market indices
        indices = [
            ("S&P 500", 4890.23, 0.45),
            ("NASDAQ", 15432.56, 0.78),
            ("DOW JONES", 38123.45, -0.23),
            ("VIX", 13.45, -2.10)
        ]
        
        for name, price, change in indices:
            change_class = "positive" if change >= 0 else "negative"
            change_sign = "+" if change >= 0 else ""
            
            st.markdown(f"""
            <div class="glass-card" style="padding: 14px 16px; margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: var(--text-secondary); font-size: 11px; font-weight: 600;">{name}</span>
                    <span class="{change_class}" style="font-size: 10px; font-family: 'JetBrains Mono';">
                        {change_sign}{change:.2f}%
                    </span>
                </div>
                <div style="font-family: 'JetBrains Mono'; font-size: 18px; font-weight: 700; color: var(--text-primary); margin-top: 4px;">
                    {price:,.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # News Section
        st.markdown("### 📰 Market News")
        
        news_items = get_market_news()
        for headline, sentiment in news_items[:5]:
            badge_class = f"badge-{sentiment.lower()}"
            
            st.markdown(f"""
            <div class="news-item">
                <span class="news-badge {badge_class}">{sentiment}</span>
                <p style="margin: 8px 0 0 0; font-size: 12px; color: var(--text-primary); line-height: 1.4;">
                    {headline}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Symbol Selection
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        symbol = st.selectbox("Select Symbol", [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM',
            'BTC-USD', 'ETH-USD', 'SOL-USD', 'SPY', 'QQQ', 'GLD', 'TQQQ'
        ], index=0)
    
    with col2:
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y"], index=1)
    
    with col3:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    # Fetch Data
    df = fetch_live_data(symbol, period=period)
    
    if not df.empty:
        df = calculate_indicators(df)
        signal, score, details = generate_ai_signal(df)
        latest_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2] if len(df) > 1 else latest_price
        price_change = latest_price - prev_price
        price_change_pct = (price_change / prev_price) * 100
        period_change = latest_price - df['Close'].iloc[0]
        period_change_pct = (period_change / df['Close'].iloc[0]) * 100
    else:
        signal, score, details = "HOLD", 50, {}
        latest_price, price_change, price_change_pct = 0, 0, 0
        period_change, period_change_pct = 0, 0
    
    # Portfolio Metrics
    metrics = calculate_portfolio_metrics(df if not df.empty else None)
    
    col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)
    
    with col_p1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Equity</div>
            <div class="metric-value">${metrics['total_equity']:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_p2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Cash Balance</div>
            <div class="metric-value">${metrics['cash']:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_p3:
        unreal_cls = "positive" if metrics['unrealized_pnl'] >= 0 else "negative"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Unrealized P&L</div>
            <div class="metric-value {unreal_cls}">${metrics['unrealized_pnl']:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_p4:
        real_cls = "positive" if metrics['realized_pnl'] >= 0 else "negative"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Realized P&L</div>
            <div class="metric-value {real_cls}">${metrics['realized_pnl']:,.0f}</div>
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
    
    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Live Trading", "🧠 Strategy Lab", "📜 Ledger"])
    
    with tab1:
        if not df.empty:
            # Chart toggles
            col_t1, col_t2 = st.columns([3, 1])
            
            with col_t1:
                show_ema20 = st.checkbox("EMA 20", value=True)
                show_ema50 = st.checkbox("EMA 50", value=True)
                show_bb = st.checkbox("Bollinger Bands", value=True)
                
                fig = create_main_chart(df, show_ema20, show_ema50, show_bb)
                st.plotly_chart(fig, use_container_width=True)
            
            with col_t2:
                st.markdown("### 🤖 AI Signal")
                
                signal_cls = f"signal-{signal.lower()}"
                st.markdown(f'<div class="signal-badge {signal_cls}">{signal}</div>', unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="glass-card" style="margin-top: 16px;">
                    <div style="text-align: center;">
                        <div style="font-family: 'JetBrains Mono'; font-size: 32px; font-weight: 700; color: var(--text-primary);">
                            {score}/100
                        </div>
                        <div style="color: var(--text-secondary); font-size: 11px; text-transform: uppercase; letter-spacing: 1px;">
                            Confidence Score
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="glass-card" style="margin-top: 12px; padding: 14px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span style="color: var(--text-secondary); font-size: 12px;">RSI (14)</span>
                        <span style="font-family: 'JetBrains Mono'; font-size: 12px; color: {'var(--accent-green)' if details.get('rsi', 50) < 50 else 'var(--accent-red)'};">
                            {details.get('rsi', 0):.1f}
                        </span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span style="color: var(--text-secondary); font-size: 12px;">Trend</span>
                        <span style="font-family: 'JetBrains Mono'; font-size: 12px; color: {'var(--accent-green)' if details.get('trend') == 'UPTREND' else 'var(--accent-red)'};">
                            {details.get('trend', 'N/A')}
                        </span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span style="color: var(--text-secondary); font-size: 12px;">MACD</span>
                        <span style="font-family: 'JetBrains Mono'; font-size: 12px; color: {'var(--accent-green)' if details.get('macd_bullish') else 'var(--accent-red)'};">
                            {"BULLISH" if details.get('macd_bullish') else "BEARISH"}
                        </span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: var(--text-secondary); font-size: 12px;">EMA Cross</span>
                        <span style="font-family: 'JetBrains Mono'; font-size: 12px; color: {'var(--accent-green)' if details.get('ema_cross') == 'above' else 'var(--accent-red)'};">
                            {details.get('ema_cross', 'N/A').upper()}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No data available for this symbol.")
    
    with tab2:
        col_s1, col_s2 = st.columns([1, 1])
        
        with col_s1:
            st.markdown("### 📊 Technical Analysis")
            
            if not df.empty:
                # Current stats
                st.markdown(f"""
                <div class="glass-card">
                    <h4 style="margin-top: 0; color: var(--text-primary);">{symbol} Analysis</h4>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
                        <div>
                            <div style="color: var(--text-secondary); font-size: 10px; text-transform: uppercase; letter-spacing: 1px;">
                                Current Price
                            </div>
                            <div style="font-family: 'JetBrains Mono'; font-size: 28px; font-weight: 700; color: var(--text-primary);">
                                ${latest_price:.2f}
                            </div>
                            <div class="{'positive' if price_change >= 0 else 'negative'}" style="font-family: 'JetBrains Mono'; font-size: 12px;">
                                {price_change:+.2f} ({price_change_pct:+.2f}%)
                            </div>
                        </div>
                        <div>
                            <div style="color: var(--text-secondary); font-size: 10px; text-transform: uppercase; letter-spacing: 1px;">
                                {period.upper()} Change
                            </div>
                            <div style="font-family: 'JetBrains Mono'; font-size: 28px; font-weight: 700; color: {'var(--accent-green)' if period_change >= 0 else 'var(--accent-red)'};">
                                {period_change:+.2f}%
                            </div>
                            <div style="color: var(--text-secondary); font-size: 12px;">
                                ${period_change:+.2f}
                            </div>
                        </div>
                    </div>
                    
                    <hr style="border-color: var(--border-color); margin: 20px 0;">
                    
                    <h5 style="color: var(--text-primary); margin-bottom: 12px;">Technical Indicators</h5>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                        <div class="glass-card" style="padding: 12px;">
                            <div style="color: var(--text-secondary); font-size: 10px;">RSI (14)</div>
                            <div style="font-family: 'JetBrains Mono'; font-size: 18px; color: {'var(--accent-green)' if details.get('rsi', 50) < 50 else 'var(--accent-red)'};">
                                {details.get('rsi', 0):.1f}
                            </div>
                        </div>
                        <div class="glass-card" style="padding: 12px;">
                            <div style="color: var(--text-secondary); font-size: 10px;">MACD</div>
                            <div style="font-family: 'JetBrains Mono'; font-size: 18px; color: {'var(--accent-green)' if details.get('macd_bullish') else 'var(--accent-red)'};">
                                {df['MACD'].iloc[-1]:+.4f}
                            </div>
                        </div>
                        <div class="glass-card" style="padding: 12px;">
                            <div style="color: var(--text-secondary); font-size: 10px;">EMA 20</div>
                            <div style="font-family: 'JetBrains Mono'; font-size: 18px;">${df['EMA_20'].iloc[-1]:.2f}</div>
                        </div>
                        <div class="glass-card" style="padding: 12px;">
                            <div style="color: var(--text-secondary); font-size: 10px;">EMA 50</div>
                            <div style="font-family: 'JetBrains Mono'; font-size: 18px;">${df['EMA_50'].iloc[-1]:.2f}</div>
                        </div>
                        <div class="glass-card" style="padding: 12px;">
                            <div style="color: var(--text-secondary); font-size: 10px;">BB Upper</div>
                            <div style="font-family: 'JetBrains Mono'; font-size: 18px;">${df['BB_Upper'].iloc[-1]:.2f}</div>
                        </div>
                        <div class="glass-card" style="padding: 12px;">
                            <div style="color: var(--text-secondary); font-size: 10px;">BB Lower</div>
                            <div style="font-family: 'JetBrains Mono'; font-size: 18px;">${df['BB_Lower'].iloc[-1]:.2f}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col_s2:
            st.markdown("### 🎯 Trading Strategy")
            
            st.markdown(f"""
            <div class="glass-card">
                <h4 style="margin-top: 0; color: var(--text-primary);">AI Recommendation</h4>
                
                <div style="text-align: center; margin: 20px 0;">
                    <span class="signal-badge signal-{signal.lower()}">{signal}</span>
                </div>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; text-align: center;">
                    <div>
                        <div style="font-family: 'JetBrains Mono'; font-size: 20px; font-weight: 700; color: {'var(--accent-green)' if signal == 'BUY' else 'var(--text-secondary)'};">BUY</div>
                        <div style="font-size: 10px; color: var(--text-secondary);">RSI &lt; 40</div>
                    </div>
                    <div>
                        <div style="font-family: 'JetBrains Mono'; font-size: 20px; font-weight: 700; color: {'var(--accent-red)' if signal == 'SELL' else 'var(--text-secondary)'};">SELL</div>
                        <div style="font-size: 10px; color: var(--text-secondary);">RSI &gt; 60</div>
                    </div>
                    <div>
                        <div style="font-family: 'JetBrains Mono'; font-size: 20px; font-weight: 700; color: {'var(--accent-gold)' if signal == 'HOLD' else 'var(--text-secondary)'};">HOLD</div>
                        <div style="font-size: 10px; color: var(--text-secondary);">Neutral</div>
                    </div>
                </div>
                
                <hr style="border-color: var(--border-color); margin: 20px 0;">
                
                <h5 style="color: var(--text-primary); margin-bottom: 12px;">Strategy Logic</h5>
                <ol style="color: var(--text-secondary); font-size: 12px; line-height: 1.8; padding-left: 18px;">
                    <li>RSI &lt; 30 indicates <span style="color: var(--accent-green);">oversold</span> → BUY signal</li>
                    <li>RSI &gt; 70 indicates <span style="color: var(--accent-red);">overbought</span> → SELL signal</li>
                    <li>EMA 20 &gt; EMA 50 → <span style="color: var(--accent-green);">Bullish</span> momentum</li>
                    <li>MACD crosses above signal → <span style="color: var(--accent-green);">Bullish</span> crossover</li>
                    <li>Price near BB Lower → <span style="color: var(--accent-green);">Support</span> zone</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        col_t1, col_t2 = st.columns([1, 2])
        
        with col_t1:
            st.markdown("### 💹 Execute Trade")
            
            with st.form("trade_form"):
                trade_type = st.radio("Trade Type", ["BUY", "SELL"], horizontal=True)
                quantity = st.number_input("Quantity", min_value=1, value=10, step=1)
                order_type = st.selectbox("Order Type", ["MARKET", "LIMIT"])
                
                if order_type == "LIMIT":
                    limit_price = st.number_input("Limit Price", value=float(latest_price), format="%.2f")
                else:
                    limit_price = latest_price
                
                total_cost = limit_price * quantity
                
                st.markdown(f"""
                <div class="glass-card" style="margin: 16px 0;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span style="color: var(--text-secondary);">Symbol</span>
                        <span style="font-family: 'JetBrains Mono';">{symbol}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span style="color: var(--text-secondary);">Price</span>
                        <span style="font-family: 'JetBrains Mono';">${limit_price:.2f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span style="color: var(--text-secondary);">Quantity</span>
                        <span style="font-family: 'JetBrains Mono';">{quantity}</span>
                    </div>
                    <hr style="border-color: var(--border-color);">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: var(--text-primary); font-weight: 600;">Total Cost</span>
                        <span style="font-family: 'JetBrains Mono'; font-weight: 700; color: {'var(--accent-red)' if trade_type == 'BUY' else 'var(--accent-green)'};">
                            ${total_cost:,.2f}
                        </span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 8px;">
                        <span style="color: var(--text-secondary);">Available Cash</span>
                        <span style="font-family: 'JetBrains Mono';">${st.session_state.virtual_balance:,.2f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.form_submit_button("📊 Execute Trade", use_container_width=True):
                    if trade_type == "BUY":
                        if total_cost <= st.session_state.virtual_balance:
                            st.session_state.virtual_balance -= total_cost
                            st.session_state.open_positions.append({
                                'symbol': symbol,
                                'quantity': quantity,
                                'entry_price': limit_price,
                                'current_price': limit_price,
                                'type': 'LONG',
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
                            })
                            st.success(f"✅ BUY executed: {quantity} {symbol} @ ${limit_price:.2f}")
                            st.rerun()
                        else:
                            st.error("❌ Insufficient funds!")
                    else:
                        # Sell logic
                        found = False
                        for i, pos in enumerate(st.session_state.open_positions):
                            if pos['symbol'] == symbol and pos['quantity'] >= quantity:
                                pnl = (limit_price - pos['entry_price']) * quantity
                                st.session_state.virtual_balance += (pos['entry_price'] * quantity) + pnl
                                st.session_state.trade_history.append({
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                                    'symbol': symbol,
                                    'type': 'SELL',
                                    'quantity': quantity,
                                    'entry': pos['entry_price'],
                                    'exit': limit_price,
                                    'pnl': pnl
                                })
                                pos['quantity'] -= quantity
                                if pos['quantity'] <= 0:
                                    st.session_state.open_positions.pop(i)
                                found = True
                                st.success(f"✅ SELL executed: {quantity} {symbol} @ ${limit_price:.2f}")
                                st.rerun()
                                break
                        
                        if not found:
                            st.error(f"❌ No {symbol} position to sell!")
            
            st.markdown("---")
            
            # Download Report
            if st.session_state.trade_history:
                df_report = pd.DataFrame(st.session_state.trade_history)
                csv = df_report.to_csv(index=False)
                st.download_button(
                    label="📥 Download Trade Logs (CSV)",
                    data=csv,
                    file_name=f"trade_logs_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col_t2:
            # Open Positions
            st.markdown("### 📁 Open Positions")
            
            if st.session_state.open_positions:
                positions_data = []
                for pos in st.session_state.open_positions:
                    current_price = latest_price if pos['symbol'] == symbol else pos['current_price']
                    pnl = (current_price - pos['entry_price']) * pos['quantity']
                    pnl_pct = ((current_price - pos['entry_price']) / pos['entry_price']) * 100
                    positions_data.append({
                        'Symbol': pos['symbol'],
                        'Qty': pos['quantity'],
                        'Entry': f"${pos['entry_price']:.2f}",
                        'Current': f"${current_price:.2f}",
                        'P&L': f"${pnl:.2f}",
                        'P&L %': f"{pnl_pct:+.2f}%"
                    })
                
                df_pos = pd.DataFrame(positions_data)
                st.dataframe(df_pos, use_container_width=True, hide_index=True)
            else:
                st.info("No open positions. Execute a trade to get started!")
            
            st.markdown("---")
            
            # Trade History
            st.markdown("### 📜 Trade History")
            
            if st.session_state.trade_history:
                df_hist = pd.DataFrame(st.session_state.trade_history)
                st.dataframe(df_hist, use_container_width=True, hide_index=True)
            else:
                st.info("No trade history yet.")


if __name__ == "__main__":
    main()
