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

st.set_page_config(
    page_title="AI Brain Pro | Trading Terminal",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    
    .stApp {
        background: #0a0e17;
    }
    
    [data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid #30363d;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #161b22 0%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    
    .metric-label {
        font-size: 10px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 24px;
        font-weight: 700;
        color: #ffffff;
    }
    
    .positive { color: #00d68f; }
    .negative { color: #ff3d71; }
    
    .live-badge {
        background: #00d68f;
        color: #000;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 10px;
        font-weight: 700;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .news-badge {
        display: inline-block;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 9px;
        font-weight: 700;
        text-transform: uppercase;
    }
    
    .badge-green { background: rgba(0, 214, 143, 0.2); color: #00d68f; }
    .badge-red { background: rgba(255, 61, 113, 0.2); color: #ff3d71; }
    .badge-gray { background: rgba(139, 148, 158, 0.2); color: #8b949e; }
    
    .glass-card {
        background: rgba(22, 27, 34, 0.9);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px;
    }
    
    .signal-box {
        padding: 16px 24px;
        border-radius: 10px;
        text-align: center;
        font-family: 'JetBrains Mono', monospace;
        font-size: 24px;
        font-weight: 700;
        letter-spacing: 2px;
    }
    
    .signal-buy { background: rgba(0, 214, 143, 0.15); border: 2px solid #00d68f; color: #00d68f; }
    .signal-sell { background: rgba(255, 61, 113, 0.15); border: 2px solid #ff3d71; color: #ff3d71; }
    .signal-hold { background: rgba(139, 148, 158, 0.15); border: 2px solid #8b949e; color: #8b949e; }
    
    .stTabs [data-baseweb="tab-list"] {
        background: #161b22;
        border-radius: 8px;
        padding: 4px;
        gap: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 6px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: #0095ff !important;
        color: white !important;
    }
    
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
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
    
    ema_cross = "above" if latest['EMA_20'] > latest['EMA_50'] else "below"
    if ema_cross == "above": score += 20
    else: score -= 20
    
    macd_bullish = latest['MACD'] > latest['MACD_Signal']
    if macd_bullish: score += 15
    else: score -= 15
    
    signal = "BUY" if score >= 65 else "SELL" if score <= 35 else "HOLD"
    
    return signal, score, {
        'rsi': rsi,
        'ema_cross': ema_cross,
        'macd_bullish': macd_bullish,
        'trend': 'UPTREND' if latest['EMA_20'] > latest['EMA_50'] else 'DOWNTREND'
    }


def get_news() -> list:
    return [
        ("Fed signals potential rate cut", "Bullish"),
        ("Tech stocks rally on earnings", "Bullish"),
        ("Oil prices stabilize", "Neutral"),
        ("Crypto markets show recovery", "Bullish"),
        ("S&P 500 reaches high", "Bullish"),
    ]


# ============================================================================
# CHART FUNCTION
# ============================================================================

def create_chart(df: pd.DataFrame, show_ema20: bool, show_ema50: bool, show_bb: bool) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2]
    )
    
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        increasing_line_color='#00d68f', decreasing_line_color='#ff3d71'
    ), row=1, col=1)
    
    if show_ema20:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], name='EMA 20', 
                                line=dict(color='#0095ff', width=1.5)), row=1, col=1)
    if show_ema50:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], name='EMA 50',
                                line=dict(color='#ffaa00', width=1.5)), row=1, col=1)
    if show_bb:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper',
                                line=dict(color='#8b949e', width=1, dash='dash'), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower',
                                line=dict(color='#8b949e', width=1, dash='dash'),
                                fill='tonexty', fillcolor='rgba(139,148,158,0.05)', showlegend=False), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='#9966ff', width=2),
                            fill='tozeroy', fillcolor='rgba(153,102,255,0.1)'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#ff3d71", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#00d68f", row=2, col=1)
    
    colors = ['#00d68f' if v >= 0 else '#ff3d71' for v in df['MACD_Hist']]
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='MACD Hist', marker_color=colors), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='#0095ff', width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='#ffaa00', width=2)), row=3, col=1)
    
    fig.update_layout(
        height=500,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        paper_bgcolor='#0a0e17',
        plot_bgcolor='#0a0e17',
        font=dict(color='#ffffff'),
        margin=dict(t=20, b=40, l=60, r=40)
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.03)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.03)')
    
    return fig


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown("### 🧠 AI Brain Pro")
    with col_h2:
        st.markdown(f"<span class='live-badge'>● LIVE</span> {datetime.now().strftime('%H:%M')}", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("#### 📊 Market Overview")
        
        indices = [("S&P 500", 4890.23, 0.45), ("NASDAQ", 15432.56, 0.78), 
                   ("DOW", 38123.45, -0.23), ("VIX", 13.45, -2.10)]
        
        for name, price, change in indices:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**{name}**")
                st.caption(f"${price:,.2f}")
            with col2:
                color = "positive" if change >= 0 else "negative"
                sign = "+" if change >= 0 else ""
                st.markdown(f"<span class='{color}'>{sign}{change:.2f}%</span>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("#### 📰 Market News")
        
        for headline, sentiment in get_news():
            color_class = "badge-green" if sentiment == "Bullish" else "badge-red" if sentiment == "Bearish" else "badge-gray"
            st.markdown(f"<span class='news-badge {color_class}'>{sentiment}</span> {headline}", unsafe_allow_html=True)
    
    # Symbol Selection
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        symbol = st.selectbox("Select Symbol", ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BTC-USD', 'ETH-USD', 'SPY', 'QQQ'])
    with col2:
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y"], index=1)
    with col3:
        if st.button("🔄 Refresh"):
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
        st.metric("Unrealized P&L", f"${unrealized_pnl:,.0f}", 
                 delta=f"{'+' if unrealized_pnl >= 0 else ''}{unrealized_pnl:,.0f}")
    with col_p4:
        st.metric("Realized P&L", f"${realized_pnl:,.0f}",
                 delta=f"{'+' if realized_pnl >= 0 else ''}{realized_pnl:,.0f}")
    with col_p5:
        st.metric("Positions", len(st.session_state.open_positions))
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Live Trading", "🧠 Strategy Lab", "📜 Ledger"])
    
    with tab1:
        if not df.empty:
            col_c1, col_c2 = st.columns([3, 1])
            
            with col_c1:
                c1, c2, c3 = st.columns(3)
                with c1:
                    show_ema20 = st.checkbox("EMA 20", value=True)
                with c2:
                    show_ema50 = st.checkbox("EMA 50", value=True)
                with c3:
                    show_bb = st.checkbox("Bollinger Bands", value=True)
                
                st.plotly_chart(create_chart(df, show_ema20, show_ema50, show_bb), use_container_width=True)
            
            with col_c2:
                st.markdown("#### 🤖 AI Signal")
                
                signal_class = f"signal-{signal.lower()}"
                st.markdown(f"<div class='signal-box {signal_class}'>{signal}</div>", unsafe_allow_html=True)
                
                st.markdown(f"**Score:** {score}/100")
                
                if details:
                    rsi_color = "positive" if details['rsi'] < 50 else "negative"
                    trend_color = "positive" if details['trend'] == 'UPTREND' else "negative"
                    macd_color = "positive" if details['macd_bullish'] else "negative"
                    
                    st.markdown("---")
                    st.markdown(f"**RSI (14):** <span class='{rsi_color}'>{details['rsi']:.1f}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Trend:** <span class='{trend_color}'>{details['trend']}</span>", unsafe_allow_html=True)
                    st.markdown(f"**MACD:** <span class='{macd_color}'>{'BULLISH' if details['macd_bullish'] else 'BEARISH'}</span>", unsafe_allow_html=True)
                    st.markdown(f"**EMA:** <span class='{trend_color}'>{details['ema_cross'].upper()}</span>", unsafe_allow_html=True)
        else:
            st.warning("No data available")
    
    with tab2:
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            st.markdown("#### 📊 Technical Analysis")
            
            if not df.empty:
                price_color = "positive" if price_change >= 0 else "negative"
                st.metric(f"{symbol} Price", f"${latest_price:.2f}", 
                         f"{'+' if price_change >= 0 else ''}{price_change_pct:.2f}%")
                
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    st.metric("RSI (14)", f"{details['rsi']:.1f}")
                    st.metric("MACD", f"{df['MACD'].iloc[-1]:+.4f}")
                with col_t2:
                    st.metric("EMA 20", f"${df['EMA_20'].iloc[-1]:.2f}")
                    st.metric("EMA 50", f"${df['EMA_50'].iloc[-1]:.2f}")
                
                col_t3, col_t4 = st.columns(2)
                with col_t3:
                    st.metric("BB Upper", f"${df['BB_Upper'].iloc[-1]:.2f}")
                with col_t4:
                    st.metric("BB Lower", f"${df['BB_Lower'].iloc[-1]:.2f}")
        
        with col_s2:
            st.markdown("#### 🎯 Trading Strategy")
            
            signal_class = f"signal-{signal.lower()}"
            st.markdown(f"<div class='signal-box {signal_class}' style='margin-bottom: 20px;'>{signal}</div>", unsafe_allow_html=True)
            
            st.markdown("**Strategy Logic:**")
            st.markdown("- RSI < 30 → **BUY** (oversold)")
            st.markdown("- RSI > 70 → **SELL** (overbought)")
            st.markdown("- EMA 20 > EMA 50 → **Bullish**")
            st.markdown("- EMA 20 < EMA 50 → **Bearish**")
            st.markdown("- MACD > Signal → **Bullish**")
    
    with tab3:
        col_t1, col_t2 = st.columns([1, 2])
        
        with col_t1:
            st.markdown("#### 💹 Execute Trade")
            
            with st.form("trade_form"):
                trade_type = st.radio("Trade Type", ["BUY", "SELL"], horizontal=True)
                quantity = st.number_input("Quantity", min_value=1, value=10)
                order_type = st.selectbox("Order Type", ["MARKET", "LIMIT"])
                
                if order_type == "LIMIT":
                    limit_price = st.number_input("Limit Price", value=float(latest_price), format="%.2f")
                else:
                    limit_price = latest_price
                
                total_cost = limit_price * quantity
                
                st.markdown(f"**Total Cost:** ${total_cost:,.2f}")
                st.markdown(f"**Available:** ${st.session_state.virtual_balance:,.2f}")
                
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
                st.download_button("📥 Download CSV", csv, f"trades_{datetime.now().strftime('%Y%m%d')}.csv", use_container_width=True)
        
        with col_t2:
            st.markdown("#### 📁 Open Positions")
            
            if st.session_state.open_positions:
                positions_data = []
                for pos in st.session_state.open_positions:
                    current_p = latest_price if pos['symbol'] == symbol else pos['current_price']
                    pnl = (current_p - pos['entry_price']) * pos['quantity']
                    pnl_pct = ((current_p - pos['entry_price']) / pos['entry_price']) * 100
                    positions_data.append({
                        'Symbol': pos['symbol'],
                        'Qty': pos['quantity'],
                        'Entry': f"${pos['entry_price']:.2f}",
                        'Current': f"${current_p:.2f}",
                        'P&L': f"${pnl:.2f} ({pnl_pct:+.1f}%)"
                    })
                st.dataframe(pd.DataFrame(positions_data), use_container_width=True, hide_index=True)
            else:
                st.info("No open positions")
            
            st.markdown("#### 📜 Trade History")
            
            if st.session_state.trade_history:
                st.dataframe(pd.DataFrame(st.session_state.trade_history), use_container_width=True, hide_index=True)
            else:
                st.info("No trade history")


if __name__ == "__main__":
    main()
