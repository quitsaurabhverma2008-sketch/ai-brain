"""
AI Brain Pro - Professional Trading Terminal v2.1
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
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    
    .stApp { background: #0a0e17; }
    
    [data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #30363d; }
    
    .stTabs [data-baseweb="tab-list"] { background: #161b22; border-radius: 12px; padding: 6px; gap: 4px; border: 1px solid #30363d; }
    .stTabs [data-baseweb="tab"] { background: transparent; border-radius: 8px; padding: 12px 24px; font-weight: 600; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #0095ff, #0070cc) !important; color: white !important; }
    
    .stButton > button { background: linear-gradient(135deg, #0095ff, #0066cc); border: none; border-radius: 10px; padding: 10px 24px; font-weight: 600; }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0, 149, 255, 0.4); }
    
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 4px; }
    
    div[data-testid="stHorizontalBlock"] > div > div { background: linear-gradient(145deg, #161b22, #0d1117); border: 1px solid #30363d; border-radius: 16px; padding: 20px 16px; margin: 4px; }
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
    return signal, score, {'rsi': rsi, 'ema_cross': "above" if latest['EMA_20'] > latest['EMA_50'] else "below",
                           'macd_bullish': latest['MACD'] > latest['MACD_Signal'],
                           'trend': 'UPTREND' if latest['EMA_20'] > latest['EMA_50'] else 'DOWNTREND'}


def generate_prediction(df: pd.DataFrame, hours: int = 24) -> list:
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


# ============================================================================
# CHART FUNCTIONS
# ============================================================================

def create_main_chart(df: pd.DataFrame, show_ema20: bool, show_ema50: bool, show_bb: bool, show_ema200: bool = False) -> go.Figure:
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.5, 0.15, 0.15, 0.2], vertical_spacing=0.03)
    
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                                name='Price', increasing_line_color='#00d68f', decreasing_line_color='#ff3d71'), row=1, col=1)
    
    if show_ema20:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], name='EMA 20', line=dict(color='#0095ff', width=2)), row=1, col=1)
    if show_ema50:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], name='EMA 50', line=dict(color='#ffaa00', width=2)), row=1, col=1)
    if show_ema200:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], name='EMA 200', line=dict(color='#8b949e', width=1.5, dash='dot')), row=1, col=1)
    if show_bb:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', line=dict(color='#9966ff', width=1, dash='dash'), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', line=dict(color='#9966ff', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(153,102,255,0.05)', showlegend=False), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='#9966ff', width=2), fill='tozeroy', fillcolor='rgba(153,102,255,0.1)'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#ff3d71", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#00d68f", row=2, col=1)
    
    colors = ['#00d68f' if v >= 0 else '#ff3d71' for v in df['MACD_Hist']]
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='MACD Hist', marker_color=colors, marker_opacity=0.8), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='#0095ff', width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='#ffaa00', width=2)), row=3, col=1)
    
    vol_colors = ['#00d68f' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#ff3d71' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=vol_colors, marker_opacity=0.6), row=4, col=1)
    
    fig.update_layout(height=600, xaxis_rangeslider_visible=False, showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                      paper_bgcolor='#0d1117', plot_bgcolor='#0d1117', font=dict(color='#ffffff'),
                      margin=dict(t=10, b=40, l=60, r=40))
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.03)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.03)')
    return fig


def create_prediction_chart(df: pd.DataFrame, predictions: list, hours: int) -> go.Figure:
    last_date = df.index[-1]
    last_price = df['Close'].iloc[-1]
    pred_dates = pd.date_range(start=last_date + timedelta(hours=1), periods=hours, freq='h')
    volatility = df['Close'].pct_change().std()
    upper_band = [p * (1 + volatility * (1 + i * 0.15)) for i, p in enumerate(predictions)]
    lower_band = [p * (1 - volatility * (1 + i * 0.15)) for i, p in enumerate(predictions)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index[-50:], y=df['Close'].tail(50), mode='lines', name='Historical', line=dict(color='#8b949e', width=2)))
    fig.add_trace(go.Scatter(x=pred_dates, y=predictions, mode='lines+markers', name='AI Prediction',
                            line=dict(color='#00d68f', width=3), marker=dict(size=8, symbol='diamond', color='#00d68f')))
    fig.add_trace(go.Scatter(x=list(pred_dates) + list(pred_dates)[::-1], y=upper_band + lower_band[::-1],
                            fill='toself', fillcolor='rgba(0, 214, 143, 0.15)', line=dict(color='rgba(0,0,0,0)'), name='Confidence Band'))
    
    fig.update_layout(height=350, xaxis_rangeslider_visible=False, showlegend=True,
                     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                     paper_bgcolor='#0d1117', plot_bgcolor='#0d1117', font=dict(color='#ffffff'),
                     margin=dict(t=10, b=40, l=60, r=40))
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.03)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.03)')
    return fig


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    col_h1, col_h2 = st.columns([4, 1])
    with col_h1:
        st.markdown("### 🧠 AI Brain Pro")
    with col_h2:
        st.markdown(f"**● LIVE** | {datetime.now().strftime('%H:%M')}")
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("#### 📊 Market Overview")
        indices = [("S&P 500", 4890.23, 0.45), ("NASDAQ", 15432.56, 0.78), ("DOW", 38123.45, -0.23), ("VIX", 13.45, -2.10)]
        for name, price, change in indices:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**{name}**")
                st.caption(f"${price:,.2f}")
            with col2:
                delta_val = change
                st.metric("", f"{'+' if delta_val >= 0 else ''}{delta_val:.2f}%", delta=delta_val)
        
        st.markdown("---")
        st.markdown("#### 📰 Market News")
        news = [("Fed signals rate cut", "Bullish"), ("Tech rally", "Bullish"), ("Oil stable", "Neutral"), ("Crypto recovery", "Bullish")]
        for headline, sent in news:
            emoji = "🟢" if sent == "Bullish" else "🔴" if sent == "Bearish" else "⚪"
            st.markdown(f"{emoji} {sent}: {headline}")
    
    # Symbol Selection
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        symbol = st.selectbox("Symbol", ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BTC-USD', 'ETH-USD', 'SPY', 'QQQ'])
    with col2:
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y"], index=1)
    with col3:
        if st.button("🔄 Refresh"):
            st.rerun()
    with col4:
        if st.button("🗑️ Reset"):
            st.session_state.virtual_balance = 100000.0
            st.session_state.open_positions = []
            st.session_state.trade_history = []
            st.rerun()
    
    # Fetch Data
    df = fetch_live_data(symbol, period=period)
    if not df.empty:
        df = calculate_indicators(df)
        signal, score, details = generate_ai_signal(df)
        latest_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2] if len(df) > 1 else latest_price
        price_change_pct = ((latest_price - prev_price) / prev_price) * 100
        period_change = ((latest_price - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
    else:
        signal, score, details = "HOLD", 50, {}
        latest_price, price_change_pct, period_change = 0, 0, 0
    
    # Portfolio Metrics
    unrealized_pnl = sum((p['current_price'] - p['entry_price']) * p['quantity'] for p in st.session_state.open_positions)
    realized_pnl = sum(t['pnl'] for t in st.session_state.trade_history)
    total_equity = st.session_state.virtual_balance + sum(p['current_price'] * p['quantity'] for p in st.session_state.open_positions)
    
    col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)
    with col_p1:
        st.metric("Total Equity", f"${total_equity:,.0f}")
    with col_p2:
        st.metric("Cash Balance", f"${st.session_state.virtual_balance:,.0f}")
    with col_p3:
        st.metric("Unrealized P&L", f"${unrealized_pnl:,.0f}", delta=f"{'+' if unrealized_pnl >= 0 else ''}{unrealized_pnl:,.0f}")
    with col_p4:
        st.metric("Realized P&L", f"${realized_pnl:,.0f}", delta=f"{'+' if realized_pnl >= 0 else ''}{realized_pnl:,.0f}")
    with col_p5:
        st.metric("Positions", len(st.session_state.open_positions))
    
    st.markdown("---")
    
    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Live Trading", "🔮 AI Prediction", "📜 Ledger"])
    
    with tab1:
        if not df.empty:
            col_c1, col_c2 = st.columns([3, 1])
            with col_c1:
                with st.expander("⚙️ Chart Settings"):
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        show_ema20 = st.checkbox("EMA 20", value=True)
                    with c2:
                        show_ema50 = st.checkbox("EMA 50", value=True)
                    with c3:
                        show_bb = st.checkbox("Bollinger Bands", value=True)
                    with c4:
                        show_ema200 = st.checkbox("EMA 200")
                
                st.plotly_chart(create_main_chart(df, show_ema20, show_ema50, show_bb, show_ema200), use_container_width=True)
            
            with col_c2:
                st.markdown("### 🤖 AI Signal")
                
                # Signal display using st.metric
                signal_emoji = "📈" if signal == "BUY" else "📉" if signal == "SELL" else "⏸️"
                st.metric("Recommendation", f"{signal_emoji} {signal}", delta=f"Score: {score}/100")
                
                st.markdown("---")
                st.markdown("#### Key Indicators")
                
                if details:
                    st.metric("RSI (14)", f"{details['rsi']:.1f}")
                    st.metric("Trend", details['trend'])
                    st.metric("MACD", "🟢 BULLISH" if details['macd_bullish'] else "🔴 BEARISH")
                    st.metric("EMA Cross", details['ema_cross'].upper())
        else:
            st.warning("No data available")
    
    with tab2:
        col_p1, col_p2 = st.columns([2, 1])
        
        with col_p1:
            st.markdown("### 📊 Price Prediction")
            
            pred_col1, pred_col2 = st.columns(2)
            with pred_col1:
                pred_time = st.selectbox("Time Horizon", ["6 Hours", "12 Hours", "24 Hours", "48 Hours", "1 Week"], index=2)
            with pred_col2:
                hours_map = {"6 Hours": 6, "12 Hours": 12, "24 Hours": 24, "48 Hours": 48, "1 Week": 168}
                hours = hours_map[pred_time]
            
            if not df.empty and len(df) > 50:
                predictions = generate_prediction(df, hours)
                pred_change = ((predictions[-1] - latest_price) / latest_price) * 100
                
                st.plotly_chart(create_prediction_chart(df, predictions, hours), use_container_width=True)
                
                stat1, stat2, stat3 = st.columns(3)
                with stat1:
                    st.metric("Current Price", f"${latest_price:.2f}")
                with stat2:
                    st.metric("Predicted Price", f"${predictions[-1]:.2f}", f"{pred_change:+.2f}%")
                with stat3:
                    st.metric("Time Horizon", pred_time)
            else:
                st.warning("Not enough data for prediction")
        
        with col_p2:
            st.markdown("### 🎯 Strategy")
            
            st.success(f"**Recommendation:** {signal} (Score: {score}/100)")
            
            st.markdown("---")
            st.markdown("#### Trading Rules:")
            st.markdown("- RSI < 30 → **BUY** (oversold)")
            st.markdown("- RSI > 70 → **SELL** (overbought)")
            st.markdown("- EMA 20 > EMA 50 → **Bullish**")
            st.markdown("- MACD > Signal → **Bullish**")
            st.markdown("- Price near BB Lower → **Support**")
    
    with tab3:
        col_t1, col_t2 = st.columns([1, 2])
        
        with col_t1:
            st.markdown("### 💹 Execute Trade")
            
            with st.form("trade_form"):
                trade_type = st.radio("Type", ["BUY", "SELL"], horizontal=True)
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
                            st.success(f"✅ BUY: {quantity} {symbol} @ ${limit_price:.2f}")
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
                                st.success("✅ SELL executed")
                                st.rerun()
                                break
                        if not found:
                            st.error("❌ No position to sell!")
            
            st.markdown("---")
            
            if st.session_state.trade_history:
                import io
                df_report = pd.DataFrame(st.session_state.trade_history)
                csv = df_report.to_csv(index=False)
                st.download_button("📥 Download CSV", csv, f"trades_{datetime.now().strftime('%Y%m%d')}.csv", use_container_width=True)
        
        with col_t2:
            st.markdown("### 📁 Open Positions")
            
            if st.session_state.open_positions:
                positions_data = []
                for pos in st.session_state.open_positions:
                    cur_p = latest_price if pos['symbol'] == symbol else pos['current_price']
                    pnl = (cur_p - pos['entry_price']) * pos['quantity']
                    pnl_pct = ((cur_p - pos['entry_price']) / pos['entry_price']) * 100
                    positions_data.append({
                        'Symbol': pos['symbol'], 'Qty': pos['quantity'],
                        'Entry': f"${pos['entry_price']:.2f}", 'Current': f"${cur_p:.2f}",
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
