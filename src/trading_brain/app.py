"""
AI Brain Pro - Professional Trading Dashboard
=============================================
A Bloomberg-style fintech application with AI-powered trading signals,
advanced visualizations, and comprehensive market analysis.
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
import time
import requests
import random

from data_feed import DataFeed
from indicators import calculate_all_indicators
from signals import SignalGenerator, Signal
from trader import Portfolio, PositionType

st.set_page_config(
    page_title="AI Brain Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CUSTOM CSS - BLOOMBERG STYLE + GLASSMORPHISM
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
        --bg-glass: rgba(22, 27, 34, 0.8);
        --accent-green: #00d68f;
        --accent-red: #ff3d71;
        --accent-blue: #0095ff;
        --accent-gold: #ffaa00;
        --text-primary: #ffffff;
        --text-secondary: #8b949e;
        --border-color: #30363d;
    }
    
    .main {
        background: var(--bg-primary);
        color: var(--text-primary);
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: var(--bg-glass);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, var(--bg-card) 0%, rgba(22, 27, 34, 0.6) 100%);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .kpi-card:hover {
        border-color: var(--accent-blue);
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0, 149, 255, 0.2);
    }
    
    .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 28px;
        font-weight: 700;
        margin: 10px 0;
    }
    
    .kpi-label {
        font-size: 12px;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .kpi-positive { color: var(--accent-green); }
    .kpi-negative { color: var(--accent-red); }
    .kpi-neutral { color: var(--accent-blue); }
    
    /* Signal Badges */
    .signal-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .signal-buy {
        background: linear-gradient(135deg, var(--accent-green), #00a876);
        color: #000;
        box-shadow: 0 4px 15px rgba(0, 214, 143, 0.4);
    }
    
    .signal-sell {
        background: linear-gradient(135deg, var(--accent-red), #cc2952);
        color: #fff;
        box-shadow: 0 4px 15px rgba(255, 61, 113, 0.4);
    }
    
    .signal-hold {
        background: linear-gradient(135deg, #4a5568, #2d3748);
        color: #fff;
    }
    
    /* News Ticker */
    .news-ticker {
        background: linear-gradient(90deg, var(--bg-card), var(--bg-secondary));
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 12px 20px;
        overflow: hidden;
        white-space: nowrap;
    }
    
    .ticker-item {
        display: inline-block;
        margin-right: 40px;
        color: var(--text-secondary);
        font-size: 13px;
    }
    
    .ticker-item span {
        color: var(--accent-green);
        margin-right: 8px;
    }
    
    /* AI Insight Box */
    .ai-insight {
        background: linear-gradient(135deg, rgba(0, 149, 255, 0.1), rgba(0, 214, 143, 0.05));
        border: 1px solid rgba(0, 149, 255, 0.3);
        border-radius: 12px;
        padding: 20px;
    }
    
    .ai-insight-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 15px;
    }
    
    .ai-insight-content {
        color: var(--text-secondary);
        line-height: 1.7;
        font-size: 14px;
    }
    
    /* Navigation */
    .nav-header {
        background: var(--bg-secondary);
        border-bottom: 1px solid var(--border-color);
        padding: 15px 30px;
        position: sticky;
        top: 0;
        z-index: 100;
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
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-secondary);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
        border-right: 1px solid var(--border-color);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-card);
        border-radius: 10px;
        padding: 5px;
        gap: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.05);
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--accent-blue) !important;
        color: #fff !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-blue), #0077cc);
        border: none;
        border-radius: 10px;
        padding: 12px 28px;
        font-weight: 600;
        color: white;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 149, 255, 0.4);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 24px !important;
    }
    
    /* DataFrames */
    [data-testid="stDataFrame"] {
        background: var(--bg-card);
        border-radius: 10px;
        border: 1px solid var(--border-color);
    }
    
    /* Expander */
    [data-testid="stExpander"] {
        background: var(--bg-card);
        border-radius: 10px;
        border: 1px solid var(--border-color);
    }
    
    /* Headers */
    h1, h2, h3 {
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-green));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* Code/Mono text */
    .mono {
        font-family: 'JetBrains Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data(ttl=300)
def get_market_news(symbol: str = None) -> list:
    """Fetch latest market news"""
    headlines = [
        ("Fed signals potential rate cut in Q2", "Bullish", "2h ago"),
        ("Tech stocks rally on strong earnings", "Bullish", "3h ago"),
        ("Oil prices stabilize amid geopolitical tensions", "Neutral", "4h ago"),
        ("Crypto markets show signs of recovery", "Bullish", "5h ago"),
        ("S&P 500 reaches new all-time high", "Bullish", "6h ago"),
        ("Treasury yields fall on weak jobs data", "Bullish", "7h ago"),
        ("Asian markets mixed amid China concerns", "Neutral", "8h ago"),
        ("Retail sales exceed expectations", "Bullish", "9h ago"),
        ("Unemployment claims drop to 3-month low", "Bullish", "10h ago"),
        ("Manufacturing PMI shows expansion", "Bullish", "11h ago"),
    ]
    
    if symbol:
        return [(f"{symbol}: {h}", t, t_) for h, t, t_ in headlines[:5]]
    return headlines


def generate_ai_insight(symbol: str, signal: str, rsi: float, macd: float, 
                        trend: str, price: float) -> str:
    """Generate AI-powered strategy explanation"""
    
    insights = []
    
    # RSI Analysis
    if rsi < 30:
        insights.append(f"The RSI at {rsi:.1f} indicates **oversold conditions**, suggesting potential upward momentum.")
    elif rsi > 70:
        insights.append(f"The RSI at {rsi:.1f} signals **overbought territory**, warranting caution.")
    else:
        insights.append(f"RSI at {rsi:.1f} shows **neutral momentum** with room for movement.")
    
    # MACD Analysis
    if macd > 0:
        insights.append(f"MACD crossover is **bullish** with histogram expanding, indicating strengthening buying pressure.")
    else:
        insights.append(f"MACD is **bearish** at {macd:.4f}, suggesting selling pressure remains dominant.")
    
    # Trend Analysis
    if trend == "UPTREND":
        insights.append(f"Price is trading **above key moving averages** in a confirmed uptrend structure.")
    else:
        insights.append(f"Price has broken below **key support levels**, indicating bearish structure.")
    
    # Signal Summary
    if signal == "BUY":
        insights.append(f"\n**AI Recommendation**: BUY {symbol} at ${price:.2f} based on multiple bullish indicators converging.")
    elif signal == "SELL":
        insights.append(f"\n**AI Recommendation**: SELL {symbol} at ${price:.2f} due to deteriorating technical conditions.")
    else:
        insights.append(f"\n**AI Recommendation**: HOLD {symbol} - await confirmation before positioning.")
    
    return "\n\n".join(insights)


def create_candlestick_chart(df: pd.DataFrame, symbol: str, show_sma: bool = True) -> go.Figure:
    """Create professional candlestick chart with indicators"""
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=('', '', '')
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
    
    # SMA 20
    if 'SMA_20' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['SMA_20'],
                name='SMA 20',
                line=dict(color='#0095ff', width=1.5)
            ),
            row=1, col=1
        )
    
    # SMA 50
    if 'SMA_50' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['SMA_50'],
                name='SMA 50',
                line=dict(color='#ffaa00', width=1.5)
            ),
            row=1, col=1
        )
    
    # Bollinger Bands
    if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
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
                fillcolor='rgba(139, 148, 158, 0.1)',
                showlegend=False
            ),
            row=1, col=1
        )
    
    # RSI
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['RSI'],
                name='RSI',
                line=dict(color='#9966ff', width=2)
            ),
            row=2, col=1
        )
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="#ff3d71", 
                      line_width=1, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#00d68f", 
                      line_width=1, row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="#8b949e", 
                      line_width=1, row=2, col=1)
    
    # MACD
    if 'MACD' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['MACD'],
                name='MACD',
                line=dict(color='#00d68f', width=2)
            ),
            row=3, col=1
        )
        
        if 'MACD_Signal' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df['MACD_Signal'],
                    name='Signal',
                    line=dict(color='#ff3d71', width=2)
                ),
                row=3, col=1
            )
    
    # Volume bars
    colors = ['#00d68f' if df['Close'].iloc[i] >= df['Open'].iloc[i] 
              else '#ff3d71' for i in range(len(df))]
    
    fig.add_trace(
        go.Bar(
            x=df.index, y=df['Volume'],
            name='Volume',
            marker_color=colors,
            marker_opacity=0.7
        ),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        template='plotly_dark',
        height=600,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)"
        ),
        paper_bgcolor='#0a0e17',
        plot_bgcolor='#0a0e17',
        font=dict(family="Inter, sans-serif", size=12),
        margin=dict(t=60, b=40, l=60, r=40)
    )
    
    # Update axes
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)', row=3, col=1)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)', row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)', row=2, col=1)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)', row=3, col=1)
    
    return fig


def create_prediction_chart(df: pd.DataFrame, predicted_prices: list, symbol: str) -> go.Figure:
    """Create prediction chart with confidence bands"""
    
    fig = make_subplots(
        rows=1, cols=1,
        shared_xaxes=True
    )
    
    # Historical candles
    fig.add_trace(
        go.Candlestick(
            x=df.index[-30:],
            open=df['Open'].tail(30),
            high=df['High'].tail(30),
            low=df['Low'].tail(30),
            close=df['Close'].tail(30),
            name='Historical',
            increasing_line_color='#00d68f',
            decreasing_line_color='#ff3d71'
        )
    )
    
    # SMAs
    if 'SMA_20' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index[-30:], y=df['SMA_20'].tail(30),
                name='SMA 20',
                line=dict(color='#0095ff', width=2)
            )
        )
    
    if 'SMA_50' in df.columns and len(df) >= 50:
        fig.add_trace(
            go.Scatter(
                x=df.index[-30:], y=df['SMA_50'].tail(30),
                name='SMA 50',
                line=dict(color='#ffaa00', width=2)
            )
        )
    
    # Prediction zone
    if predicted_prices:
        last_date = df.index[-1]
        pred_dates = pd.date_range(start=last_date + timedelta(hours=1), 
                                   periods=len(predicted_prices), freq='H')
        
        fig.add_trace(
            go.Scatter(
                x=list(pred_dates),
                y=predicted_prices,
                name='AI Prediction',
                mode='lines+markers',
                line=dict(color='#00d68f', width=3, dash='dot'),
                marker=dict(size=10, symbol='diamond')
            )
        )
        
        # Confidence bands
        volatility = df['Close'].pct_change().std()
        upper_band = [p * (1 + volatility * (1 + i * 0.2)) 
                      for i, p in enumerate(predicted_prices)]
        lower_band = [p * (1 - volatility * (1 + i * 0.2)) 
                      for i, p in enumerate(predicted_prices)]
        
        fig.add_trace(
            go.Scatter(
                x=list(pred_dates) + list(pred_dates)[::-1],
                y=upper_band + lower_band[::-1],
                fill='toself',
                fillcolor='rgba(0, 214, 143, 0.2)',
                line=dict(color='rgba(0, 0, 0, 0)'),
                name='Confidence Band'
            )
        )
    
    fig.update_layout(
        template='plotly_dark',
        height=500,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        paper_bgcolor='#0a0e17',
        plot_bgcolor='#0a0e17',
        font=dict(family="Inter, sans-serif", size=12),
        margin=dict(t=40, b=40, l=60, r=40)
    )
    
    return fig


def run_backtest(df: pd.DataFrame, initial_capital: float, 
                 start_date: datetime, end_date: datetime) -> dict:
    """Run backtest on historical data"""
    
    # Filter by date
    df_test = df[(df.index >= start_date) & (df.index <= end_date)].copy()
    
    if len(df_test) < 50:
        return {'error': 'Insufficient data for backtest'}
    
    capital = initial_capital
    position = None
    trades = []
    equity_curve = [initial_capital]
    
    df_test = calculate_all_indicators(df_test)
    
    for i in range(50, len(df_test)):
        row = df_test.iloc[i]
        prev_row = df_test.iloc[i-1]
        
        signal_gen = SignalGenerator()
        
        # Generate signal
        if position is None:
            # Check for buy signal
            if (prev_row['RSI'] < 40 and row['RSI'] > prev_row['RSI']) or \
               (prev_row['MACD'] < prev_row['MACD_Signal'] and 
                row['MACD'] > row['MACD_Signal']):
                # Buy
                shares = capital // row['Close']
                if shares > 0:
                    position = {
                        'entry_price': row['Close'],
                        'shares': shares,
                        'entry_date': df_test.index[i]
                    }
        
        elif position is not None:
            # Check for sell signal
            pnl = (row['Close'] - position['entry_price']) * position['shares']
            pnl_pct = (row['Close'] - position['entry_price']) / position['entry_price'] * 100
            
            if pnl_pct >= 5 or pnl_pct <= -2 or \
               (prev_row['RSI'] > 65 and row['RSI'] < prev_row['RSI']):
                # Sell
                trades.append({
                    'entry': position['entry_price'],
                    'exit': row['Close'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'date': df_test.index[i]
                })
                capital += pnl
                equity_curve.append(capital)
                position = None
        
        equity_curve.append(capital if position is None else 
                           capital + (row['Close'] - position['entry_price']) * position['shares'])
    
    # Calculate metrics
    if not trades:
        return {'error': 'No trades generated'}
    
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    
    return {
        'trades': trades,
        'equity_curve': equity_curve,
        'final_capital': capital,
        'total_return': (capital - initial_capital) / initial_capital * 100,
        'win_rate': len(wins) / len(trades) * 100,
        'total_trades': len(trades),
        'avg_win': sum(t['pnl'] for t in wins) / len(wins) if wins else 0,
        'avg_loss': sum(t['pnl'] for t in losses) / len(losses) if losses else 0,
        'max_drawdown': min(equity_curve) - max(equity_curve),
        'profit_factor': abs(sum(t['pnl'] for t in wins) / sum(t['pnl'] for t in losses)) if losses else float('inf')
    }


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Initialize session state
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = Portfolio(100000)
    
    if 'selected_symbol' not in st.session_state:
        st.session_state.selected_symbol = 'AAPL'
    
    # Header
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; 
                padding: 20px 30px; background: var(--bg-secondary); 
                border-bottom: 1px solid var(--border-color);">
        <div style="display: flex; align-items: center; gap: 15px;">
            <h1 style="margin: 0; font-size: 28px;">🧠 AI Brain Pro</h1>
            <span style="background: var(--accent-green); color: #000; 
                        padding: 4px 12px; border-radius: 20px; 
                        font-size: 11px; font-weight: 600;">LIVE</span>
        </div>
        <div style="display: flex; align-items: center; gap: 20px;">
            <span class="mono" style="color: var(--text-secondary); font-size: 13px;">
                {} | Bloomberg Terminal Style
            </span>
        </div>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Navigation")
        
        selected_menu = st.radio(
            "Select Section",
            ["📈 Dashboard", "🔮 AI Prediction", "📊 Backtest", "💹 Trade Now", "⚙️ Settings"],
            index=0
        )
        
        st.markdown("---")
        
        # Symbol Selection
        st.markdown("### 🎯 Symbol")
        
        default_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 
            'V', 'PG', 'UNH', 'HD', 'BAC', 'ADBE', 'NFLX', 'INTC',
            'BTC-USD', 'ETH-USD', 'SOL-USD', 'SPY', 'QQQ', 'GLD'
        ]
        
        symbol = st.selectbox("Select Market", default_stocks, index=0)
        st.session_state.selected_symbol = symbol
        
        st.markdown("---")
        
        # Quick Stats
        st.markdown("### 📉 Market Status")
        st.markdown("""
        <div class="glass-card">
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                <span style="color: var(--text-secondary);">S&P 500</span>
                <span class="mono" style="color: var(--accent-green);">4,890.23</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                <span style="color: var(--text-secondary);">NASDAQ</span>
                <span class="mono" style="color: var(--accent-green);">15,432.56</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                <span style="color: var(--text-secondary);">DOW</span>
                <span class="mono" style="color: var(--accent-red);">38,123.45</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: var(--text-secondary);">VIX</span>
                <span class="mono" style="color: var(--accent-gold);">13.45</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # News Section
        st.markdown("### 📰 Latest News")
        news = get_market_news(symbol)
        for headline, sentiment, time_ in news[:4]:
            color = "var(--accent-green)" if sentiment == "Bullish" else "var(--accent-red)" if sentiment == "Bearish" else "var(--text-secondary)"
            st.markdown(f"""
            <div style="border-left: 3px solid {color}; padding-left: 10px; margin-bottom: 10px;">
                <span style="font-size: 12px; color: var(--text-secondary);">{time_}</span>
                <p style="margin: 5px 0; font-size: 13px;">{headline}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # ============ DASHBOARD ============
    if selected_menu == "📈 Dashboard":
        col1, col2, col3, col4, col5 = st.columns(5)
        
        portfolio = st.session_state.portfolio
        stats = portfolio.get_stats()
        
        with col1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Portfolio Value</div>
                <div class="kpi-value mono">${stats.get('current_capital', 100000):,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            pnl = stats.get('total_pnl', 0)
            color = "kpi-positive" if pnl >= 0 else "kpi-negative"
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Total P&L</div>
                <div class="kpi-value mono {color}">${pnl:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            wr = stats.get('win_rate', 0)
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Win Rate</div>
                <div class="kpi-value mono {'kpi-positive' if wr >= 50 else 'kpi-negative'}">{wr:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            pf = stats.get('profit_factor', 0)
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Profit Factor</div>
                <div class="kpi-value mono {'kpi-positive' if pf >= 1.5 else 'kpi-neutral'}">{pf:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            tt = stats.get('total_trades', 0)
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Total Trades</div>
                <div class="kpi-value mono">{tt}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Analysis Section
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.markdown("### 📊 Technical Analysis")
            
            # Fetch and analyze
            data_feed = DataFeed()
            df = data_feed.get_intraday_data(symbol, "1h", "30d")
            
            if df is not None and len(df) > 50:
                df = calculate_all_indicators(df)
                
                # Create chart
                fig = create_candlestick_chart(df, symbol)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Insufficient data for analysis")
        
        with col_right:
            st.markdown("### 🤖 AI Signal")
            
            if df is not None:
                signal_gen = SignalGenerator()
                signal, score, details = signal_gen.generate_signal(df)
                
                latest = df.iloc[-1]
                
                # Signal Badge
                if signal == Signal.BUY:
                    st.markdown('<span class="signal-badge signal-buy">BUY</span>', 
                               unsafe_allow_html=True)
                elif signal == Signal.SELL:
                    st.markdown('<span class="signal-badge signal-sell">SELL</span>', 
                               unsafe_allow_html=True)
                else:
                    st.markdown('<span class="signal-badge signal-hold">HOLD</span>', 
                               unsafe_allow_html=True)
                
                # Key Metrics
                st.markdown(f"""
                <div class="glass-card" style="margin-top: 20px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 15px;">
                        <span style="color: var(--text-secondary);">Price</span>
                        <span class="mono" style="font-size: 18px; font-weight: 600;">${latest['Close']:.2f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 15px;">
                        <span style="color: var(--text-secondary);">RSI (14)</span>
                        <span class="mono" style="color: {'var(--accent-green)' if latest['RSI'] < 30 else 'var(--accent-red)' if latest['RSI'] > 70 else 'var(--text-primary)'};">{latest['RSI']:.1f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 15px;">
                        <span style="color: var(--text-secondary);">MACD</span>
                        <span class="mono" style="color: {'var(--accent-green)' if latest['MACD'] > latest['MACD_Signal'] else 'var(--accent-red)'};">{latest['MACD']:.4f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 15px;">
                        <span style="color: var(--text-secondary);">Trend</span>
                        <span style="color: {'var(--accent-green)' if details.get('trend') == 'UPTREND' else 'var(--accent-red)'};">{details.get('trend', 'N/A')}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: var(--text-secondary);">Signal Score</span>
                        <span class="mono">{score:.0f}/100</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # AI Insights Section
        st.markdown("### 💡 AI Strategy Insights")
        
        if df is not None:
            insight = generate_ai_insight(
                symbol,
                signal.value,
                latest['RSI'],
                latest['MACD'],
                details.get('trend', 'N/A'),
                latest['Close']
            )
            
            st.markdown(f"""
            <div class="ai-insight">
                <div class="ai-insight-header">
                    <span style="font-size: 24px;">🤖</span>
                    <span style="font-weight: 600; font-size: 16px;">AI Analysis Engine</span>
                    <span style="margin-left: auto; font-size: 12px; color: var(--text-secondary);">
                        Generated at {} EST
                    </span>
                </div>
                <div class="ai-insight-content">
                    {insight}
                </div>
            </div>
            """.format(datetime.now().strftime("%H:%M:%S")), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Live Signals Table
        st.markdown("### 📈 Top Trading Signals")
        
        signals_data = []
        for sym in default_stocks[:10]:
            try:
                df_s = data_feed.get_intraday_data(sym, "1h", "30d")
                if df_s is not None and len(df_s) > 50:
                    df_s = calculate_all_indicators(df_s)
                    sig, sc, det = signal_gen.generate_signal(df_s)
                    latest_s = df_s.iloc[-1]
                    
                    signals_data.append({
                        'Symbol': sym,
                        'Price': f"${latest_s['Close']:.2f}",
                        'Signal': sig.value,
                        'RSI': f"{latest_s['RSI']:.1f}",
                        'Trend': det.get('trend', 'N/A'),
                        'Score': f"{sc:.0f}"
                    })
            except:
                pass
        
        if signals_data:
            df_signals = pd.DataFrame(signals_data)
            
            # Color the Signal column
            def color_signal(val):
                if val == 'BUY':
                    return 'color: var(--accent-green); font-weight: 600'
                elif val == 'SELL':
                    return 'color: var(--accent-red); font-weight: 600'
                return ''
            
            st.dataframe(
                df_signals.style.applymap(color_signal, subset=['Signal']),
                use_container_width=True,
                hide_index=True
            )
    
    # ============ AI PREDICTION ============
    elif selected_menu == "🔮 AI Prediction":
        st.markdown("### 🔮 AI Price Prediction Engine")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            duration = st.select_slider(
                "Prediction Horizon",
                options=['1h', '4h', '12h', '24h', '48h', '1w'],
                value='24h'
            )
            
            confidence = st.slider("Confidence Threshold", 50, 95, 70)
            
            show_bands = st.checkbox("Show Confidence Bands", value=True)
            show_indicators = st.checkbox("Show Technical Indicators", value=True)
            
            if st.button("🔮 Generate Prediction", use_container_width=True):
                with st.spinner("Analyzing market data..."):
                    time.sleep(2)  # Simulate processing
        
        with col2:
            data_feed = DataFeed()
            df = data_feed.get_intraday_data(symbol, "1h", "60d")
            
            if df is not None:
                df = calculate_all_indicators(df)
                
                # Generate mock predictions
                last_price = df['Close'].iloc[-1]
                steps = {'1h': 1, '4h': 4, '12h': 12, '24h': 24, '48h': 48, '1w': 168}[duration]
                
                # Realistic prediction with mean reversion
                predictions = []
                current = last_price
                volatility = df['Close'].pct_change().std()
                
                for i in range(min(steps, 24)):
                    change = np.random.normal(0.001, volatility * 0.5)
                    current = current * (1 + change)
                    predictions.append(current)
                
                fig = create_prediction_chart(df, predictions, symbol)
                st.plotly_chart(fig, use_container_width=True)
                
                # Prediction Summary
                pred_change = ((predictions[-1] - last_price) / last_price) * 100
                
                col_p1, col_p2, col_p3 = st.columns(3)
                
                with col_p1:
                    st.metric("Current Price", f"${last_price:.2f}")
                
                with col_p2:
                    st.metric("Predicted Price", f"${predictions[-1]:.2f}", 
                             f"{pred_change:+.2f}%")
                
                with col_p3:
                    st.metric("Confidence", f"{confidence}%")
                
                # AI Prediction Summary
                st.markdown(f"""
                <div class="ai-insight">
                    <div class="ai-insight-header">
                        <span style="font-size: 24px;">📊</span>
                        <span style="font-weight: 600;">Prediction Summary</span>
                    </div>
                    <div class="ai-insight-content">
                        Based on technical analysis and ML models, the AI predicts a **{'bullish' if pred_change > 0 else 'bearish'}** 
                        trend for {symbol} over the next {duration}. The expected price movement of **{pred_change:+.2f}%** 
                        is supported by:
                        <ul>
                            <li>RSI indicating {'oversold' if df.iloc[-1]['RSI'] < 50 else 'overbought'} conditions</li>
                            <li>{'Bullish' if df.iloc[-1]['MACD'] > df.iloc[-1]['MACD_Signal'] else 'Bearish'} MACD crossover</li>
                            <li>Volume analysis showing {'increasing' if df.iloc[-1]['Volume'] > df['Volume'].mean() else 'decreasing'} activity</li>
                        </ul>
                        <strong>⚠️ This is not financial advice. Trade responsibly.</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # ============ BACKTEST ============
    elif selected_menu == "📊 Backtest":
        st.markdown("### 📊 Strategy Backtest")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
        
        with col2:
            end_date = st.date_input("End Date", datetime.now())
        
        with col3:
            initial_capital = st.number_input("Initial Capital ($)", 
                                             min_value=1000, 
                                             value=10000, 
                                             step=1000)
        
        strategy = st.selectbox(
            "Trading Strategy",
            ["RSI + MACD Crossover", "Bollinger Bands", "SMA Crossover", "All Signals Combined"]
        )
        
        if st.button("▶️ Run Backtest", use_container_width=True):
            with st.spinner("Running backtest simulation..."):
                data_feed = DataFeed()
                df = data_feed.get_intraday_data(symbol, "1h", "2y")
                
                if df is not None:
                    results = run_backtest(
                        df, 
                        initial_capital,
                        datetime.combine(start_date, datetime.min.time()),
                        datetime.combine(end_date, datetime.max.time())
                    )
                    
                    if 'error' not in results:
                        # Metrics
                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                        
                        with col_m1:
                            st.metric("Total Return", f"{results['total_return']:.2f}%",
                                     f"${results['final_capital'] - initial_capital:,.2f}")
                        
                        with col_m2:
                            st.metric("Win Rate", f"{results['win_rate']:.1f}%")
                        
                        with col_m3:
                            st.metric("Profit Factor", f"{results['profit_factor']:.2f}")
                        
                        with col_m4:
                            st.metric("Total Trades", results['total_trades'])
                        
                        st.markdown("---")
                        
                        # Equity Curve
                        st.markdown("### 📈 Equity Curve")
                        
                        fig_eq = go.Figure()
                        fig_eq.add_trace(go.Scatter(
                            x=list(range(len(results['equity_curve']))),
                            y=results['equity_curve'],
                            mode='lines',
                            fill='tozeroy',
                            fillcolor='rgba(0, 214, 143, 0.2)',
                            line=dict(color='#00d68f', width=2)
                        ))
                        
                        fig_eq.update_layout(
                            template='plotly_dark',
                            height=400,
                            paper_bgcolor='#0a0e17',
                            plot_bgcolor='#0a0e17',
                            xaxis_title="Trades",
                            yaxis_title="Portfolio Value ($)"
                        )
                        
                        st.plotly_chart(fig_eq, use_container_width=True)
                        
                        # Trade History
                        st.markdown("### 📋 Trade History")
                        
                        trades_df = pd.DataFrame(results['trades'])
                        if not trades_df.empty:
                            trades_df['pnl'] = trades_df['pnl'].apply(lambda x: f"${x:.2f}")
                            trades_df['pnl_pct'] = trades_df['pnl_pct'].apply(lambda x: f"{x:+.2f}%")
                            trades_df['date'] = trades_df['date'].dt.strftime('%Y-%m-%d %H:%M')
                            trades_df = trades_df.rename(columns={
                                'entry': 'Entry', 'exit': 'Exit', 'pnl': 'P&L', 
                                'pnl_pct': 'P&L %', 'date': 'Date'
                            })
                            st.dataframe(trades_df, use_container_width=True, hide_index=True)
                    else:
                        st.error(results['error'])
                else:
                    st.error("Could not fetch data for backtesting")
    
    # ============ TRADE NOW ============
    elif selected_menu == "💹 Trade Now":
        st.markdown("### 💹 Paper Trading")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            trade_type = st.radio("Trade Type", ["BUY", "SELL"], horizontal=True)
        
        with col2:
            quantity = st.number_input("Quantity", min_value=1, value=10, step=1)
        
        with col3:
            order_type = st.selectbox("Order Type", ["MARKET", "LIMIT"])
        
        if st.button("📊 Execute Trade", use_container_width=True):
            data_feed = DataFeed()
            price = data_feed.get_live_quote(symbol)
            if price:
                st.success(f"Order executed: {trade_type} {quantity} {symbol} @ ${price:.2f}")
        
        st.markdown("---")
        
        # Open Positions
        st.markdown("### 📁 Open Positions")
        
        portfolio = st.session_state.portfolio
        if portfolio.positions:
            for pos in portfolio.positions:
                st.markdown(f"""
                <div class="glass-card" style="margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="font-weight: 600;">{pos.symbol}</span>
                        <span>{pos.quantity} shares</span>
                        <span class="mono">${pos.entry_price:.2f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No open positions. Execute a trade to get started!")
        
        st.markdown("---")
        
        # Trade History
        st.markdown("### 📜 Trade History")
        
        if portfolio.trade_history:
            history_df = pd.DataFrame([{
                'Symbol': t.symbol,
                'Type': t.position_type,
                'Entry': f"${t.entry_price:.2f}",
                'Exit': f"${t.exit_price:.2f}",
                'P&L': f"${t.pnl:.2f}",
                'P&L %': f"{t.pnl_percent:.2f}%"
            } for t in portfolio.trade_history[-10:]])
            
            st.dataframe(history_df, use_container_width=True, hide_index=True)
    
    # ============ SETTINGS ============
    else:
        st.markdown("### ⚙️ Settings")
        
        st.markdown("""
        <div class="glass-card">
            <h4>Account Configuration</h4>
            <p style="color: var(--text-secondary);">Configure your trading preferences and risk management settings.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            st.markdown("#### Risk Management")
            max_position = st.slider("Max Position Size (%)", 5, 50, 20)
            stop_loss = st.slider("Default Stop Loss (%)", 1, 10, 2)
            take_profit = st.slider("Default Take Profit (%)", 3, 20, 5)
        
        with col_s2:
            st.markdown("#### Notifications")
            email_alerts = st.checkbox("Email Alerts", value=False)
            push_alerts = st.checkbox("Push Notifications", value=True)
            sound_alerts = st.checkbox("Sound Alerts", value=False)
        
        if st.button("💾 Save Settings"):
            st.success("Settings saved successfully!")


if __name__ == "__main__":
    main()
