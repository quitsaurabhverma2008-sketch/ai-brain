"""
AI Brain Pro - Professional Trading Terminal v2.0
================================================
Bloomberg-style trading terminal with AI signals, prediction, and paper trading.
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

st.set_page_config(
    page_title="AI Brain Pro | Trading Terminal",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - PROFESSIONAL BLOOMBERG STYLE
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .stApp { background: #0a0e17; }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        border-bottom: 1px solid #30363d;
        padding: 16px 24px;
        margin: -1rem -1rem 1.5rem -1rem;
        border-radius: 0;
    }
    
    .brand-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 26px;
        font-weight: 700;
        background: linear-gradient(135deg, #0095ff, #00d68f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .live-badge {
        background: #00d68f;
        color: #000;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #0a0e17 100%);
        border-right: 1px solid #30363d;
        padding-top: 1rem;
    }
    
    /* Metric Cards */
    div[data-testid="stHorizontalBlock"] > div > div {
        background: linear-gradient(145deg, #161b22 0%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 20px 16px;
        margin: 4px;
        transition: all 0.3s ease;
    }
    
    div[data-testid="stHorizontalBlock"] > div > div:hover {
        border-color: #0095ff;
        box-shadow: 0 0 30px rgba(0, 149, 255, 0.1);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #161b22;
        border-radius: 12px;
        padding: 6px;
        gap: 4px;
        border: 1px solid #30363d;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 14px;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.05);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0095ff, #0070cc) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(0, 149, 255, 0.3);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0095ff, #0066cc);
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 149, 255, 0.4);
    }
    
    /* Selectbox */
    [data-testid="stSelectbox"] [data-baseweb="select"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
    }
    
    /* Checkbox */
    .stCheckbox > label {
        background: #161b22;
        padding: 8px 16px;
        border-radius: 8px;
        border: 1px solid #30363d;
        font-size: 13px;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #161b22;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    
    /* DataFrame */
    [data-testid="stDataFrame"] {
        background: #161b22;
        border-radius: 12px;
        border: 1px solid #30363d;
    }
    
    /* Success/Error messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 10px;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #484f58; }
    
    /* News badges */
    .badge-bullish {
        background: rgba(0, 214, 143, 0.2);
        color: #00d68f;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
    }
    
    .badge-bearish {
        background: rgba(255, 61, 113, 0.2);
        color: #ff3d71;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
    }
    
    .badge-neutral {
        background: rgba(139, 148, 158, 0.2);
        color: #8b949e;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
    }
    
    /* Signal cards */
    .signal-card {
        background: #161b22;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        border: 2px solid;
    }
    
    .signal-buy {
        border-color: #00d68f;
        box-shadow: 0 0 40px rgba(0, 214, 143, 0.2);
    }
    
    .signal-sell {
        border-color: #ff3d71;
        box-shadow: 0 0 40px rgba(255, 61, 113, 0.2);
    }
    
    .signal-hold {
        border-color: #8b949e;
    }
    
    /* Market index cards */
    .index-card {
        background: linear-gradient(145deg, #161b22, #0d1117);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }
    
    /* Chart container */
    .chart-container {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 16px;
    }
    
    /* Strategy card */
    .strategy-card {
        background: linear-gradient(145deg, #161b22 0%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE
# ============================================================================

if 'virtual_balance' not in st.session_state:
    st.session_state.virtual_balance = 100000.0
if 'open_positions' not in st.session_state:
    st.session_state.open_positions = []
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []


# ============================================================================
# DATA FUNCTIONS
# ============================================================================

@st.cache_data(ttl=60)
def fetch_live_data(symbol: str, period: str = "3mo", interval: str = "1h") -> pd.DataFrame:
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except:
        return pd.DataFrame()


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df


def generate_ai_signal(df: pd.DataFrame) -> tuple:
    if len(df) < 50:
        return "HOLD", 50, {}
    
    latest = df.iloc[-1]
    score = 50
    
    rsi = latest['RSI']
    if rsi < 30: score += 25
    elif rsi < 40: score += 15
    elif rsi > 70: score -= 25
    elif rsi > 60: score -= 15
    
    if latest['EMA_20'] > latest['EMA_50']: score += 20
    else: score -= 20
    
    if latest['MACD'] > latest['MACD_Signal']: score += 15
    else: score -= 15
    
    signal = "BUY" if score >= 65 else "SELL" if score <= 35 else "HOLD"
    
    return signal, score, {
        'rsi': rsi,
        'ema_cross': "above" if latest['EMA_20'] > latest['EMA_50'] else "below",
        'macd_bullish': latest['MACD'] > latest['MACD_Signal'],
        'trend': 'UPTREND' if latest['EMA_20'] > latest['EMA_50'] else 'DOWNTREND'
    }


def generate_prediction(df: pd.DataFrame, hours: int = 24) -> tuple:
    last_price = df['Close'].iloc[-1]
    volatility = df['Close'].pct_change().std()
    trend = (df['EMA_20'].iloc[-1] - df['EMA_50'].iloc[-1]) / df['EMA_50'].iloc[-1]
    
    predictions = []
    current = last_price
    
    for i in range(hours):
        bias = 1 + (trend * (i / hours))
        change = np.random.normal(bias * 0.001, volatility * 0.5)
        current = current * (1 + change)
        predictions.append(current)
    
    return predictions


def get_news() -> list:
    return [
        ("Fed signals potential rate cut", "Bullish"),
        ("Tech stocks rally on earnings", "Bullish"),
        ("Oil prices stabilize", "Neutral"),
        ("Crypto markets recovery", "Bullish"),
        ("S&P 500 reaches high", "Bullish"),
    ]


# ============================================================================
# CHART FUNCTIONS
# ============================================================================

def create_main_chart(df: pd.DataFrame, show_ema20: bool, show_ema50: bool, show_bb: bool, show_ema200: bool = False) -> go.Figure:
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        vertical_spacing=0.03
    )
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Price', increasing_line_color='#00d68f', decreasing_line_color='#ff3d71',
        increasing_fillcolor='rgba(0, 214, 143, 0.3)', decreasing_fillcolor='rgba(255, 61, 113, 0.3)'
    ), row=1, col=1)
    
    if show_ema20:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], name='EMA 20', 
                                line=dict(color='#0095ff', width=2)), row=1, col=1)
    if show_ema50:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], name='EMA 50',
                                line=dict(color='#ffaa00', width=2)), row=1, col=1)
    if show_ema200:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], name='EMA 200',
                                line=dict(color='#8b949e', width=1.5, dash='dot')), row=1, col=1)
    if show_bb:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper',
                                line=dict(color='#9966ff', width=1, dash='dash'), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower',
                                line=dict(color='#9966ff', width=1, dash='dash'), fill='tonexty',
                                fillcolor='rgba(153, 102, 255, 0.05)', showlegend=False), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                            line=dict(color='#9966ff', width=2),
                            fill='tozeroy', fillcolor='rgba(153, 102, 255, 0.1)'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#ff3d71", line_width=1, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#00d68f", line_width=1, row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="#8b949e", line_width=1, row=2, col=1)
    
    # MACD
    colors = ['#00d68f' if v >= 0 else '#ff3d71' for v in df['MACD_Hist']]
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='MACD Hist',
                        marker_color=colors, marker_opacity=0.8), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                            line=dict(color='#0095ff', width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal',
                            line=dict(color='#ffaa00', width=2)), row=3, col=1)
    
    # Volume
    vol_colors = ['#00d68f' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#ff3d71' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume',
                        marker_color=vol_colors, marker_opacity=0.6), row=4, col=1)
    
    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                    bgcolor="rgba(0,0,0,0)", font=dict(color='#8b949e', size=11)),
        paper_bgcolor='#0d1117',
        plot_bgcolor='#0d1117',
        font=dict(color='#ffffff', family="Inter, sans-serif", size=11),
        margin=dict(t=10, b=40, l=60, r=40)
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.03)', showticklabels=True)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.03)')
    
    return fig


def create_prediction_chart(df: pd.DataFrame, predictions: list, hours: int) -> go.Figure:
    last_date = df.index[-1]
    last_price = df['Close'].iloc[-1]
    pred_dates = pd.date_range(start=last_date + timedelta(hours=1), periods=hours, freq='h')
    
    # Calculate confidence bands
    volatility = df['Close'].pct_change().std()
    upper_band = [p * (1 + volatility * (1 + i * 0.15)) for i, p in enumerate(predictions)]
    lower_band = [p * (1 - volatility * (1 + i * 0.15)) for i, p in enumerate(predictions)]
    
    fig = go.Figure()
    
    # Historical price
    fig.add_trace(go.Scatter(
        x=df.index[-50:],
        y=df['Close'].tail(50),
        mode='lines',
        name='Historical',
        line=dict(color='#8b949e', width=2)
    ))
    
    # Prediction line
    fig.add_trace(go.Scatter(
        x=pred_dates,
        y=predictions,
        mode='lines+markers',
        name='AI Prediction',
        line=dict(color='#00d68f', width=3),
        marker=dict(size=8, symbol='diamond', color='#00d68f')
    ))
    
    # Confidence band
    fig.add_trace(go.Scatter(
        x=list(pred_dates) + list(pred_dates)[::-1],
        y=upper_band + lower_band[::-1],
        fill='toself',
        fillcolor='rgba(0, 214, 143, 0.15)',
        line=dict(color='rgba(0, 0, 0, 0)'),
        name='Confidence Band'
    ))
    
    # Upper/Lower bounds
    fig.add_trace(go.Scatter(
        x=pred_dates, y=upper_band, mode='lines',
        line=dict(color='#00d68f', width=1, dash='dash'),
        name='Upper Bound', opacity=0.5
    ))
    fig.add_trace(go.Scatter(
        x=pred_dates, y=lower_band, mode='lines',
        line=dict(color='#ff3d71', width=1, dash='dash'),
        name='Lower Bound', opacity=0.5
    ))
    
    fig.update_layout(
        height=350,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        paper_bgcolor='#0d1117',
        plot_bgcolor='#0d1117',
        font=dict(color='#ffffff'),
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
    <div class="main-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 class="brand-title">🧠 AI Brain Pro</h1>
            </div>
            <div style="display: flex; align-items: center; gap: 20px;">
                <span class="live-badge">● LIVE</span>
                <span style="color: #8b949e; font-family: 'JetBrains Mono'; font-size: 12px;">
                    {} EST
                </span>
            </div>
        </div>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Market Overview")
        
        indices = [
            ("S&P 500", 4890.23, 0.45),
            ("NASDAQ", 15432.56, 0.78),
            ("DOW JONES", 38123.45, -0.23),
            ("VIX", 13.45, -2.10)
        ]
        
        for name, price, change in indices:
            color = "#00d68f" if change >= 0 else "#ff3d71"
            sign = "+" if change >= 0 else ""
            st.markdown(f"""
            <div class="index-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #8b949e; font-size: 11px;">{name}</span>
                    <span style="color: {color}; font-size: 10px; font-family: 'JetBrains Mono';">{sign}{change:.2f}%</span>
                </div>
                <div style="font-family: 'JetBrains Mono'; font-size: 20px; font-weight: 700; color: #fff; margin-top: 4px;">
                    {price:,.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📰 Market News")
        
        for headline, sentiment in get_news():
            badge = f"badge-{'bullish' if sentiment == 'Bullish' else 'bearish' if sentiment == 'Bearish' else 'neutral'}"
            st.markdown(f"<span class='{badge}'>{sentiment}</span> {headline}", unsafe_allow_html=True)
    
    # Symbol Selection
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
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
    
    with col4:
        if st.button("🗑️ Reset Portfolio", use_container_width=True):
            st.session_state.virtual_balance = 100000.0
            st.session_state.open_positions = []
            st.session_state.trade_history = []
            st.success("Portfolio reset!")
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
        period_change = ((latest_price - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
    else:
        signal, score, details = "HOLD", 50, {}
        latest_price, price_change, price_change_pct, period_change = 0, 0, 0, 0
    
    # Portfolio Metrics
    unrealized_pnl = sum((pos['current_price'] - pos['entry_price']) * pos['quantity'] for pos in st.session_state.open_positions)
    realized_pnl = sum(t['pnl'] for t in st.session_state.trade_history)
    total_equity = st.session_state.virtual_balance + sum(pos['current_price'] * pos['quantity'] for pos in st.session_state.open_positions)
    
    col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)
    
    with col_p1:
        st.metric("Total Equity", f"${total_equity:,.0f}")
    with col_p2:
        st.metric("Cash Balance", f"${st.session_state.virtual_balance:,.0f}")
    with col_p3:
        delta_color = "normal" if unrealized_pnl >= 0 else "inverse"
        st.metric("Unrealized P&L", f"${unrealized_pnl:,.0f}", 
                 delta=f"{'+' if unrealized_pnl >= 0 else ''}{unrealized_pnl:,.0f}")
    with col_p4:
        st.metric("Realized P&L", f"${realized_pnl:,.0f}",
                 delta=f"{'+' if realized_pnl >= 0 else ''}{realized_pnl:,.0f}")
    with col_p5:
        st.metric("Positions", len(st.session_state.open_positions))
    
    st.markdown("---")
    
    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Live Trading", "🔮 AI Prediction", "📜 Ledger"])
    
    with tab1:
        if not df.empty:
            col_c1, col_c2 = st.columns([3, 1])
            
            with col_c1:
                # Chart toggles
                with st.expander("⚙️ Chart Settings", expanded=False):
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        show_ema20 = st.checkbox("EMA 20", value=True)
                    with c2:
                        show_ema50 = st.checkbox("EMA 50", value=True)
                    with c3:
                        show_bb = st.checkbox("Bollinger Bands", value=True)
                    with c4:
                        show_ema200 = st.checkbox("EMA 200", value=False)
                
                st.plotly_chart(create_main_chart(df, show_ema20, show_ema50, show_bb, show_ema200), use_container_width=True)
            
            with col_c2:
                st.markdown("### 🤖 AI Signal")
                
                signal_class = f"signal-{signal.lower()}"
                st.markdown(f"<div class='signal-card {signal_class}'><div style='font-size: 32px; font-weight: 700; font-family: JetBrains Mono;'>{signal}</div><div style='color: #8b949e; margin-top: 8px;'>Confidence: {abs(score-50)*2}%</div></div>", unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown("#### Key Indicators")
                
                if details:
                    rsi_val = details['rsi']
                    rsi_color = "#00d68f" if rsi_val < 50 else "#ff3d71"
                    trend_color = "#00d68f" if details['trend'] == 'UPTREND' else "#ff3d71"
                    macd_color = "#00d68f" if details['macd_bullish'] else "#ff3d71"
                    
                    st.markdown(f"""
                    <div style="background: #161b22; border-radius: 12px; padding: 16px; border: 1px solid #30363d;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                            <span style="color: #8b949e;">RSI (14)</span>
                            <span style="font-family: 'JetBrains Mono'; color: {rsi_color};">{rsi_val:.1f}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                            <span style="color: #8b949e;">Trend</span>
                            <span style="font-family: 'JetBrains Mono'; color: {trend_color};">{details['trend']}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                            <span style="color: #8b949e;">MACD</span>
                            <span style="font-family: 'JetBrains Mono'; color: {macd_color};">{'BULLISH' if details['macd_bullish'] else 'BEARISH'}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #8b949e;">EMA Cross</span>
                            <span style="font-family: 'JetBrains Mono'; color: {trend_color};">{details['ema_cross'].upper()}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No data available for this symbol")
    
    with tab2:
        col_p1, col_p2 = st.columns([2, 1])
        
        with col_p1:
            st.markdown("### 📊 Price Prediction")
            
            # Prediction time selector
            pred_col1, pred_col2, pred_col3 = st.columns(3)
            with pred_col1:
                pred_hours = st.selectbox("Prediction Time", ["6 Hours", "12 Hours", "24 Hours", "48 Hours", "1 Week"], index=2)
            with pred_col2:
                hours_map = {"6 Hours": 6, "12 Hours": 12, "24 Hours": 24, "48 Hours": 48, "1 Week": 168}
                hours = hours_map[pred_hours]
            
            if not df.empty and len(df) > 50:
                predictions = generate_prediction(df, hours)
                pred_change = ((predictions[-1] - latest_price) / latest_price) * 100
                
                st.plotly_chart(create_prediction_chart(df, predictions, hours), use_container_width=True)
                
                # Prediction stats
                stat_col1, stat_col2, stat_col3 = st.columns(3)
                with stat_col1:
                    st.metric("Current Price", f"${latest_price:.2f}")
                with stat_col2:
                    st.metric("Predicted Price", f"${predictions[-1]:.2f}", f"{pred_change:+.2f}%")
                with stat_col3:
                    st.metric("Time Horizon", pred_hours)
            else:
                st.warning("Not enough data for prediction")
        
        with col_p2:
            st.markdown("### 🎯 Strategy")
            
            st.markdown(f"""
            <div class="strategy-card">
                <h4 style="margin-top: 0; color: #fff;">AI Recommendation</h4>
                <div style="text-align: center; margin: 20px 0;">
                    <span class="badge-{'bullish' if signal == 'BUY' else 'bearish' if signal == 'SELL' else 'neutral'}" style="font-size: 14px; padding: 8px 20px;">
                        {signal}
                    </span>
                </div>
                
                <hr style="border-color: #30363d; margin: 20px 0;">
                
                <h5 style="color: #fff; margin-bottom: 12px;">Trading Rules</h5>
                <ol style="color: #8b949e; font-size: 12px; line-height: 1.8; padding-left: 18px;">
                    <li>RSI &lt; 30 → <span style="color: #00d68f;">BUY</span> (oversold)</li>
                    <li>RSI &gt; 70 → <span style="color: #ff3d71;">SELL</span> (overbought)</li>
                    <li>EMA 20 &gt; EMA 50 → <span style="color: #00d68f;">Bullish</span></li>
                    <li>MACD &gt; Signal → <span style="color: #00d68f;">Bullish</span> crossover</li>
                    <li>Price near BB Lower → <span style="color: #00d68f;">Support</span></li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        col_t1, col_t2 = st.columns([1, 2])
        
        with col_t1:
            st.markdown("### 💹 Execute Trade")
            
            with st.form("trade_form"):
                trade_type = st.radio("Trade Type", ["BUY", "SELL"], horizontal=True)
                quantity = st.number_input("Quantity", min_value=1, value=10)
                order_type = st.selectbox("Order Type", ["MARKET", "LIMIT"])
                
                if order_type == "LIMIT":
                    limit_price = st.number_input("Limit Price", value=float(latest_price), format="%.2f")
                else:
                    limit_price = latest_price
                
                total_cost = limit_price * quantity
                
                st.markdown(f"""
                <div style="background: #161b22; border-radius: 12px; padding: 16px; margin: 16px 0; border: 1px solid #30363d;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span style="color: #8b949e;">Symbol</span>
                        <span style="font-family: 'JetBrains Mono';">{symbol}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span style="color: #8b949e;">Price</span>
                        <span style="font-family: 'JetBrains Mono';">${limit_price:.2f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span style="color: #8b949e;">Quantity</span>
                        <span style="font-family: 'JetBrains Mono';">{quantity}</span>
                    </div>
                    <hr style="border-color: #30363d;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #fff; font-weight: 600;">Total Cost</span>
                        <span style="font-family: 'JetBrains Mono'; font-weight: 700; color: #ff3d71;">${total_cost:,.2f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 8px;">
                        <span style="color: #8b949e;">Available Cash</span>
                        <span style="font-family: 'JetBrains Mono';">${st.session_state.virtual_balance:,.2f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.form_submit_button("📊 Execute Trade", use_container_width=True):
                    if trade_type == "BUY":
                        if total_cost <= st.session_state.virtual_balance:
                            st.session_state.virtual_balance -= total_cost
                            st.session_state.open_positions.append({
                                'symbol': symbol, 'quantity': quantity,
                                'entry_price': limit_price, 'current_price': limit_price,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
                            })
                            st.success(f"✅ BUY executed: {quantity} {symbol} @ ${limit_price:.2f}")
                            st.rerun()
                        else:
                            st.error("❌ Insufficient funds!")
                    else:
                        found = False
                        for i, pos in enumerate(st.session_state.open_positions):
                            if pos['symbol'] == symbol and pos['quantity'] >= quantity:
                                pnl = (limit_price - pos['entry_price']) * quantity
                                st.session_state.virtual_balance += (pos['entry_price'] * quantity) + pnl
                                st.session_state.trade_history.append({
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                                    'symbol': symbol, 'type': 'SELL', 'quantity': quantity,
                                    'entry': pos['entry_price'], 'exit': limit_price, 'pnl': pnl
                                })
                                pos['quantity'] -= quantity
                                if pos['quantity'] <= 0:
                                    st.session_state.open_positions.pop(i)
                                found = True
                                st.success(f"✅ SELL executed")
                                st.rerun()
                                break
                        if not found:
                            st.error("❌ No position to sell!")
            
            st.markdown("---")
            
            if st.session_state.trade_history:
                df_report = pd.DataFrame(st.session_state.trade_history)
                csv = df_report.to_csv(index=False)
                st.download_button("📥 Download Trade Logs", csv, 
                                 f"trades_{datetime.now().strftime('%Y%m%d')}.csv",
                                 use_container_width=True)
        
        with col_t2:
            st.markdown("### 📁 Open Positions")
            
            if st.session_state.open_positions:
                positions_data = []
                for pos in st.session_state.open_positions:
                    cur_p = latest_price if pos['symbol'] == symbol else pos['current_price']
                    pnl = (cur_p - pos['entry_price']) * pos['quantity']
                    pnl_pct = ((cur_p - pos['entry_price']) / pos['entry_price']) * 100
                    positions_data.append({
                        'Symbol': pos['symbol'],
                        'Qty': pos['quantity'],
                        'Entry': f"${pos['entry_price']:.2f}",
                        'Current': f"${cur_p:.2f}",
                        'P&L': f"${pnl:.2f} ({pnl_pct:+.1f}%)"
                    })
                st.dataframe(pd.DataFrame(positions_data), use_container_width=True, hide_index=True)
            else:
                st.info("No open positions")
            
            st.markdown("---")
            st.markdown("### 📜 Trade History")
            
            if st.session_state.trade_history:
                st.dataframe(pd.DataFrame(st.session_state.trade_history), use_container_width=True, hide_index=True)
            else:
                st.info("No trade history")


if __name__ == "__main__":
    main()
