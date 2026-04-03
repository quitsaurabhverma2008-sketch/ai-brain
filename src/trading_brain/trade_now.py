"""
Trading Brain - Trade Now Module
=================================
Live trading simulation with AI signals
"""

import sys
import os
# Add src directory to path
src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_dir)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

# Page config
st.set_page_config(page_title="Trade Now", page_icon="🎯", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .buy-box { 
        background: linear-gradient(135deg, #00d4aa 0%, #00a878 100%); 
        padding: 25px; 
        border-radius: 16px; 
        text-align: center;
        box-shadow: 0 8px 30px rgba(0, 212, 170, 0.3);
    }
    .sell-box { 
        background: linear-gradient(135deg, #ff4757 0%, #c0392b 100%); 
        padding: 25px; 
        border-radius: 16px; 
        text-align: center;
        box-shadow: 0 8px 30px rgba(255, 71, 87, 0.3);
    }
    .hold-box { 
        background: linear-gradient(135deg, #5a5a7a 0%, #3d3d5c 100%); 
        padding: 25px; 
        border-radius: 16px; 
        text-align: center;
    }
    .signal-display { font-size: 32px; font-weight: bold; }
    .trade-timer { font-size: 24px; color: #ffa500; }
    .pnl-positive { color: #00d4aa; font-size: 20px; font-weight: 600; }
    .pnl-negative { color: #ff4757; font-size: 20px; font-weight: 600; }
    
    /* Headers */
    h1, h2, h3 {
        background: linear-gradient(90deg, #00d4aa, #00a878);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


class PaperTrader:
    """Paper trading simulator"""
    
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = []
        self.trade_history = []
    
    def open_trade(self, symbol, entry_price, quantity, trade_type="LONG"):
        """Open a paper trade"""
        cost = entry_price * quantity
        if cost > self.cash:
            return False, "Insufficient funds"
        
        trade = {
            'symbol': symbol,
            'entry_time': datetime.now(),
            'entry_price': entry_price,
            'quantity': quantity,
            'type': trade_type,
            'sl_price': entry_price * 0.97,
            'tp_price': entry_price * 1.08,
            'status': 'OPEN',
            'exit_time': None,
            'exit_price': None,
            'pnl': 0,
            'pnl_pct': 0
        }
        
        self.positions.append(trade)
        self.cash -= cost
        return True, f"Opened {trade_type} {quantity} shares at ${entry_price:.2f}"
    
    def close_trade(self, index, exit_price, reason):
        """Close a paper trade"""
        if index >= len(self.positions):
            return False, "Position not found"
        
        trade = self.positions[index]
        
        if trade['type'] == 'LONG':
            pnl = (exit_price - trade['entry_price']) * trade['quantity']
        else:
            pnl = (trade['entry_price'] - exit_price) * trade['quantity']
        
        pnl_pct = (pnl / (trade['entry_price'] * trade['quantity'])) * 100
        
        trade['status'] = 'CLOSED'
        trade['exit_time'] = datetime.now()
        trade['exit_price'] = exit_price
        trade['pnl'] = pnl
        trade['pnl_pct'] = pnl_pct
        trade['exit_reason'] = reason
        
        self.trade_history.append(trade)
        self.cash += exit_price * trade['quantity']
        
        del self.positions[index]
        return True, f"Closed at ${exit_price:.2f} | P&L: ${pnl:.2f} ({pnl_pct:.1f}%)"
    
    def check_auto_exits(self, current_price):
        """Check if any positions hit SL/TP"""
        closed = []
        for i, trade in enumerate(self.positions[:]):
            if trade['type'] == 'LONG':
                if current_price <= trade['sl_price']:
                    reason = 'STOP_LOSS'
                elif current_price >= trade['tp_price']:
                    reason = 'TAKE_PROFIT'
                else:
                    reason = None
            else:
                if current_price >= trade['sl_price']:
                    reason = 'STOP_LOSS'
                elif current_price <= trade['tp_price']:
                    reason = 'TAKE_PROFIT'
                else:
                    reason = None
            
            if reason:
                success, msg = self.close_trade(i, current_price, reason)
                if success:
                    closed.append(msg)
        return closed
    
    def get_stats(self):
        """Get trading statistics"""
        if not self.trade_history:
            return {
                'trades': 0, 
                'wins': 0, 
                'win_rate': 0, 
                'total_pnl': 0, 
                'capital': self.cash,
                'open_positions': len(self.positions)
            }
        
        wins = sum(1 for t in self.trade_history if t['pnl'] > 0)
        total = len(self.trade_history)
        
        return {
            'trades': total,
            'wins': wins,
            'win_rate': (wins/total*100) if total > 0 else 0,
            'total_pnl': sum(t['pnl'] for t in self.trade_history),
            'capital': self.cash,
            'open_positions': len(self.positions)
        }


def create_live_chart(df, symbol, trade_entry=None):
    """Create live price chart with indicators"""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2])
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name='Price'
    ), row=1, col=1)
    
    # SMA 20
    if 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20',
                               line=dict(color='#00cc66', width=1)), row=1, col=1)
    
    # SMA 50
    if 'SMA_50' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50',
                               line=dict(color='#ff6b6b', width=1)), row=1, col=1)
    
    # Bollinger Bands
    if 'BB_Upper' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper',
                               line=dict(color='#666', width=1), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower',
                               line=dict(color='#666', width=1), fill='tonexty',
                               fillcolor='rgba(255,255,255,0.05)', showlegend=False), row=1, col=1)
    
    # Trade markers
    if trade_entry:
        fig.add_vline(x=trade_entry, line_dash="dash", line_color="#00cc66",
                     annotation_text="ENTRY", row=1, col=1)
    
    # RSI
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                               line=dict(color='#ffa500', width=2)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#ff4b4b", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#00cc66", row=2, col=1)
    
    # MACD
    if 'MACD' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                               line=dict(color='#00ccff', width=2)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal',
                               line=dict(color='#ffa500', width=2)), row=3, col=1)
    
    fig.update_layout(template='plotly_dark', height=500,
                     xaxis_rangeslider_visible=False,
                     title=f'{symbol} - Live Chart')
    
    return fig


def main():
    # Initialize session state
    if 'paper_trader' not in st.session_state:
        st.session_state.paper_trader = PaperTrader(100000)
    
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = True
    
    trader = st.session_state.paper_trader
    
    # Sidebar
    st.sidebar.title("🎯 Trade Now")
    
    # Stock selection
    symbol = st.sidebar.selectbox("Select Stock", 
                                   ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'SPY'])
    
    # Timeframe
    interval = st.sidebar.selectbox("Timeframe", ['15m', '1h', '4h'], index=1)
    
    # Manual refresh
    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()
    
    # Auto refresh toggle
    st.session_state.auto_refresh = st.sidebar.checkbox("Auto-refresh (10s)", value=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Portfolio")
    stats = trader.get_stats()
    st.sidebar.metric("Capital", f"${stats['capital']:,.0f}")
    st.sidebar.metric("Open Positions", stats['open_positions'])
    st.sidebar.metric("Total P&L", f"${stats['total_pnl']:,.0f}", 
                     delta=f"{stats['win_rate']:.0f}% win rate")
    
    # Main
    st.title("🎯 Trade Now - Live Trading")
    
    # Load data
    with st.spinner(f"Loading {symbol} data..."):
        from data_feed import DataFeed
        from indicators import calculate_all_indicators
        from signals import SignalGenerator
        from ml_model import MLTradingModel
        
        feed = DataFeed()
        df = feed.get_intraday_data(symbol, interval, "60d")
        
        if len(df) > 50:
            df = calculate_all_indicators(df)
        
        gen = SignalGenerator()
        quote = feed.get_live_quote(symbol)
        
        # ML Model Prediction
        ml_result = None
        if len(df) > 50:
            try:
                ml_model = MLTradingModel("models")
                if ml_model.load_model():
                    ml_result = ml_model.analyze_current(df)
                else:
                    # Try to use simplified prediction if no model
                    ml_result = {
                        'direction': 'HOLD',
                        'confidence': 0,
                        'ml_signal': 'HOLD'
                    }
            except Exception as e:
                ml_result = {
                    'direction': 'HOLD',
                    'confidence': 0,
                    'ml_signal': 'HOLD'
                }
        else:
            ml_result = {'direction': 'HOLD', 'confidence': 0, 'ml_signal': 'HOLD'}
    
    # Current price
    if quote:
        current_price = quote['price']
    else:
        current_price = df.iloc[-1]['Close'] if len(df) > 0 else 0
    
    # Get AI signal
    if len(df) > 50:
        signal, score, details = gen.generate_signal(df)
        rsi = details.get('rsi', 50)
        trend = details.get('trend', 'N/A')
    else:
        signal = None
        score = 0
        details = {}
        rsi = 50
        trend = 'N/A'
    
    # ===== SIGNALS SECTION =====
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Price chart
        trade_entry_time = None
        if trader.positions:
            trade_entry_time = trader.positions[0]['entry_time']
        
        fig = create_live_chart(df, symbol, trade_entry_time)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🧠 AI Brain Response")
        
        # Combined Signal Analysis
        st.markdown("**📊 Technical Analysis**")
        
        if signal:
            # Signal box
            if signal.value == 'BUY':
                st.markdown(f'''
                <div class="buy-box">
                    <div class="signal-display">🟢 BUY</div>
                    <div>Score: {score}</div>
                    <div>RSI: {rsi:.1f}</div>
                    <div>Trend: {trend}</div>
                </div>
                ''', unsafe_allow_html=True)
                
                # Entry levels
                levels = gen.get_entry_levels(current_price)
                st.markdown(f"""
                **Entry:** ${current_price:.2f}
                
                **Stop Loss:** ${levels['stop_loss']:.2f} (-3%)
                **Take Profit:** ${levels['take_profit']:.2f} (+8%)
                
                **Risk/Reward:** {levels['risk_reward_ratio']}:1
                """)
                
                # Open trade button
                if st.button("🟢 OPEN TRADE", type="primary", use_container_width=True):
                    quantity = int((trader.cash * 0.10) / current_price)  # 10% per trade
                    if quantity > 0:
                        success, msg = trader.open_trade(symbol, current_price, quantity, "LONG")
                        if success:
                            st.success(msg)
                        else:
                            st.error(msg)
                    else:
                        st.error("Insufficient funds")
            
            elif signal.value == 'SELL':
                st.markdown(f'''
                <div class="sell-box">
                    <div class="signal-display">🔴 SELL</div>
                    <div>Score: {score}</div>
                    <div>RSI: {rsi:.1f}</div>
                    <div>Trend: {trend}</div>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="hold-box">
                    <div class="signal-display">⏸️ HOLD</div>
                    <div>Score: {score}</div>
                    <div>RSI: {rsi:.1f}</div>
                    <div>Trend: {trend}</div>
                </div>
                ''', unsafe_allow_html=True)
        
        # ML Model Signal
        st.markdown("---")
        st.markdown("**🤖 ML Prediction**")
        
        if ml_result:
            ml_confidence = ml_result.get('confidence', 0)
            ml_signal = ml_result.get('ml_signal', 'HOLD')
            
            # Color based on signal
            if ml_signal == 'BUY':
                ml_color = "#00cc66"
            elif ml_signal == 'SELL':
                ml_color = "#ff4b4b"
            else:
                ml_color = "#ffa500"
            
            st.markdown(f"""
            <div style="background: {ml_color}20; padding: 15px; border-radius: 10px; border: 1px solid {ml_color};">
                <div style="font-size: 24px; font-weight: bold; color: {ml_color};">
                    {ml_signal} ({ml_confidence:.1f}% confidence)
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Combined signal check
            st.markdown("---")
            st.markdown("**🎯 Combined Signal**")
            
            if signal and ml_result:
                tech_signal = signal.value
                combined = "STRONG BUY" if (tech_signal == 'BUY' and ml_signal == 'BUY') else \
                           "STRONG SELL" if (tech_signal == 'SELL' and ml_signal == 'SELL') else \
                           "BUY" if (tech_signal == 'BUY' or ml_signal == 'BUY') else \
                           "SELL" if (tech_signal == 'SELL' or ml_signal == 'SELL') else "HOLD"
                
                if combined in ["STRONG BUY", "BUY"]:
                    combined_color = "#00cc66"
                elif combined in ["STRONG SELL", "SELL"]:
                    combined_color = "#ff4b4b"
                else:
                    combined_color = "#ffa500"
                
                st.markdown(f"""
                <div style="background: {combined_color}30; padding: 15px; border-radius: 10px; border: 2px solid {combined_color};">
                    <div style="font-size: 28px; font-weight: bold; color: {combined_color};">
                        {combined}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Current price
        st.markdown("---")
        st.markdown(f"### 💰 Current Price")
        if quote:
            st.metric(f"{symbol}", f"${current_price:.2f}", 
                      delta=f"{quote['change_pct']:.2f}%")
        else:
            st.metric(f"{symbol}", f"${current_price:.2f}")
    
    # ===== OPEN POSITIONS =====
    st.markdown("---")
    st.markdown("### 📋 Open Positions")
    
    if trader.positions:
        for i, pos in enumerate(trader.positions):
            # Calculate current P&L
            if pos['type'] == 'LONG':
                pnl = (current_price - pos['entry_price']) * pos['quantity']
                pnl_pct = ((current_price - pos['entry_price']) / pos['entry_price']) * 100
            else:
                pnl = (pos['entry_price'] - current_price) * pos['quantity']
                pnl_pct = ((pos['entry_price'] - current_price) / pos['entry_price']) * 100
            
            # Time elapsed
            elapsed = datetime.now() - pos['entry_time']
            hours = elapsed.total_seconds() / 3600
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.markdown(f"**{pos['symbol']}** {pos['type']}")
                st.caption(f"Entry: ${pos['entry_price']:.2f}")
            
            with col2:
                st.markdown(f"${current_price:.2f}")
                st.caption("Current")
            
            with col3:
                st.markdown(f"${pos['sl_price']:.2f} / ${pos['tp_price']:.2f}")
                st.caption("SL / TP")
            
            with col4:
                if pnl >= 0:
                    st.markdown(f'<p class="pnl-positive">+${pnl:.2f} ({pnl_pct:.1f}%)</p>', 
                               unsafe_allow_html=True)
                else:
                    st.markdown(f'<p class="pnl-negative">${pnl:.2f} ({pnl_pct:.1f}%)</p>', 
                               unsafe_allow_html=True)
            
            with col5:
                if st.button(f"Close", key=f"close_{i}"):
                    success, msg = trader.close_trade(i, current_price, "MANUAL")
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
    else:
        st.info("No open positions")
    
    # ===== TRADE HISTORY =====
    st.markdown("---")
    st.markdown("### 📜 Trade History")
    
    if trader.trade_history:
        history_df = pd.DataFrame([{
            'Symbol': t['symbol'],
            'Type': t['type'],
            'Entry': f"${t['entry_price']:.2f}",
            'Exit': f"${t['exit_price']:.2f}" if t['exit_price'] else "Open",
            'P&L': f"${t['pnl']:.2f}",
            'P&L %': f"{t['pnl_pct']:.1f}%",
            'Reason': t.get('exit_reason', 'N/A'),
            'Duration': f"{((t['exit_time'] - t['entry_time']).total_seconds()/3600):.1f}h" 
                       if t['exit_time'] else "Open"
        } for t in reversed(trader.trade_history[-10:])])
        
        st.dataframe(history_df, use_container_width=True)
        
        # Stats
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Trades", stats['trades'])
        col2.metric("Win Rate", f"{stats['win_rate']:.1f}%")
        col3.metric("Winners", stats['wins'])
        col4.metric("Total P&L", f"${stats['total_pnl']:,.0f}")
    else:
        st.info("No trades yet")
    
    # Auto refresh
    if st.session_state.auto_refresh:
        # Check auto exits
        if trader.positions and quote:
            closed = trader.check_auto_exits(current_price)
            if closed:
                for msg in closed:
                    st.success(msg)
        
        time.sleep(10)
        st.rerun()


if __name__ == "__main__":
    main()


def show_trade_page():
    """Wrapper to show trade page from dashboard"""
    main()
