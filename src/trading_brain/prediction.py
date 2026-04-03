"""
AI Prediction Page
=================
Generate AI-powered price predictions with entry/exit levels
And future predicted candles
"""

import sys
import os
src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_dir)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Page config
st.set_page_config(page_title="AI Prediction", page_icon="📈", layout="wide")

# CSS for modern professional styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    /* Prediction Boxes */
    .up-box { 
        background: linear-gradient(135deg, #00d4aa 0%, #00a878 100%); 
        padding: 25px; 
        border-radius: 16px; 
        box-shadow: 0 8px 30px rgba(0, 212, 170, 0.3);
    }
    .down-box { 
        background: linear-gradient(135deg, #ff4757 0%, #c0392b 100%); 
        padding: 25px; 
        border-radius: 16px; 
        box-shadow: 0 8px 30px rgba(255, 71, 87, 0.3);
    }
    .entry-box { 
        border: 2px solid #00d4aa; 
        padding: 20px; 
        border-radius: 16px; 
        background: linear-gradient(135deg, #1a1a24 0%, #16161e 100%);
    }
    .sl-box { 
        border: 2px solid #ff4757; 
        padding: 20px; 
        border-radius: 16px; 
        background: linear-gradient(135deg, #1a1a24 0%, #16161e 100%);
    }
    .tp-box { 
        border: 2px solid #00d4aa; 
        padding: 20px; 
        border-radius: 16px; 
        background: linear-gradient(135deg, #1a1a24 0%, #16161e 100%);
    }
    .confidence-high { 
        color: #00d4aa; 
        font-size: 28px; 
        font-weight: bold; 
    }
    .confidence-low { 
        color: #ffa500; 
        font-size: 28px; 
        font-weight: bold; 
    }
    .future-label { 
        background: linear-gradient(90deg, rgba(0,212,170,0.15), transparent);
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 14px;
        color: #00d4aa;
        font-weight: 600;
    }
    .duration-btn {
        background: linear-gradient(135deg, #1a1a24, #16161e);
        border: 1px solid #2a2a3a;
        border-radius: 12px;
        padding: 15px;
        margin: 5px;
        transition: all 0.3s ease;
    }
    .duration-btn:hover {
        border-color: #00d4aa;
        transform: translateY(-2px);
    }
    
    /* Headers */
    h1, h2, h3 {
        background: linear-gradient(90deg, #00d4aa, #00a878);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# All stocks list
ALL_STOCKS = [
    # US Stocks
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'JPM', 'JNJ',
    'V', 'PG', 'UNH', 'MA', 'HD', 'DIS', 'PYPL', 'BAC', 'ADBE', 'CRM',
    'NFLX', 'INTC', 'VZ', 'T', 'XOM', 'KO', 'PEP', 'WMT', 'ABT', 'MRK',
    'CVX', 'LLY', 'PFE', 'ABBV', 'TMO', 'COST', 'AVGO', 'NEE', 'DHR', 'NKE',
    'TXN', 'QCOM', 'HON', 'UPS', 'PM', 'MS', 'GS', 'BLK', 'IBM', 'AMD',
    'CAT', 'BA', 'MMM', 'RTX', 'LOW', 'SBUX', 'SCHW', 'BKNG', 'ISRG', 'MDLZ',
    # Crypto
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 
    'AVAX-USD', 'DOGE-USD', 'DOT-USD', 'MATIC-USD',
    # ETFs
    'SPY', 'QQQ', 'IWM', 'GLD', 'TLT'
]

# Duration options with more realistic timeframes
DURATION_OPTIONS = [
    ('15m', '15 Minutes', 1),
    ('30m', '30 Minutes', 2),
    ('1h', '1 Hour', 4),
    ('2h', '2 Hours', 8),
    ('4h', '4 Hours', 16),
    ('8h', '8 Hours', 32),
    ('1d', '1 Day', 48),
    ('2d', '2 Days', 96)
]

# Duration options
DURATION_OPTIONS = [
    ('15m', '15 Minutes', 1),
    ('30m', '30 Minutes', 2),
    ('1h', '1 Hour', 4),
    ('4h', '4 Hours', 16),
    ('24h', '24 Hours', 24)
]


def get_data(symbol, interval, period):
    """Fetch stock data"""
    import yfinance as yf
    
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(interval=interval, period=period)
        
        if df.empty:
            return None
        
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        return None


def calculate_indicators(df):
    """Calculate technical indicators"""
    from indicators import calculate_all_indicators
    
    if df is None or len(df) < 50:
        return None
    
    try:
        df = calculate_all_indicators(df)
        return df
    except:
        return None


def get_ml_prediction(df, num_steps=24):
    """Get ML model prediction with future prices"""
    from ml_model import MLTradingModel
    
    if df is None or len(df) < 50:
        return {'direction': 'UNKNOWN', 'confidence': 0, 'future_prices': []}
    
    try:
        model = MLTradingModel("models")
        
        if not model.load_model():
            return {'direction': 'UNKNOWN', 'confidence': 0, 'fallback': True, 'future_prices': []}
        
        # Get current analysis
        current_result = model.analyze_current(df)
        
        # Get future price predictions with more steps
        future_result = model.predict_future_prices(df, steps=num_steps)
        
        # Combine results
        return {
            'direction': future_result.get('direction', 'HOLD'),
            'confidence': future_result.get('confidence', 0),
            'prob_up': future_result.get('prob_up', 50),
            'prob_down': future_result.get('prob_down', 50),
            'future_prices': future_result.get('prices', []),
            'avg_change': future_result.get('avg_change', 0),
            'volatility': future_result.get('volatility', 1),
            'trend': future_result.get('trend', 'N/A'),
            'rsi': future_result.get('rsi', 50),
            'tech_bullish': future_result.get('tech_bullish', 0),
            'tech_bearish': future_result.get('tech_bearish', 0)
        }
    except Exception as e:
        return {'direction': 'UNKNOWN', 'confidence': 0, 'error': str(e), 'future_prices': []}


def get_technical_signal(df):
    """Get technical analysis signal"""
    from signals import SignalGenerator
    
    if df is None or len(df) < 50:
        return {'signal': 'HOLD', 'score': 0}
    
    try:
        gen = SignalGenerator()
        signal, score, details = gen.generate_signal(df)
        
        return {
            'signal': signal.value,
            'score': score,
            'rsi': details.get('rsi', 50),
            'trend': details.get('trend', 'N/A'),
            'details': details
        }
    except:
        return {'signal': 'HOLD', 'score': 0}


def calculate_levels(current_price, future_prices, direction, confidence):
    """Calculate entry, SL, TP levels based on predictions"""
    
    # Use predicted prices for TP if available
    if future_prices:
        predicted_price = future_prices[-1] if future_prices else current_price
    else:
        predicted_price = current_price
    
    # Risk parameters
    sl_pct = 0.02
    tp_pct = 0.06
    
    # Adjust based on confidence
    if confidence > 70:
        sl_pct = 0.015
        tp_pct = 0.08
    elif confidence < 50:
        sl_pct = 0.025
        tp_pct = 0.04
    
    if direction == 'UP':
        entry = current_price
        sl = current_price * (1 - sl_pct)
        tp = predicted_price * 1.02 if predicted_price > current_price else current_price * (1 + tp_pct)
    elif direction == 'DOWN':
        entry = current_price
        sl = current_price * (1 + sl_pct)
        tp = predicted_price * 0.98 if predicted_price < current_price else current_price * (1 - tp_pct)
    else:
        entry = current_price
        sl = current_price * 0.98
        tp = current_price * 1.05
    
    risk_reward = abs((tp - entry) / (entry - sl)) if entry != sl else 0
    
    return {
        'entry': round(entry, 2),
        'sl': round(sl, 2),
        'tp': round(tp, 2),
        'sl_pct': sl_pct * 100,
        'tp_pct': tp_pct * 100,
        'risk_reward': round(risk_reward, 2)
    }


def create_prediction_chart(df, symbol, current_price, levels, direction, confidence, 
                           interval, predicted_prices, duration_label, ml_result=None):
    """Create professional candlestick chart with predicted future candles"""
    
    if df is None or len(df) < 20:
        return None
    
    # Take last 30 candles for historical data
    chart_df = df.tail(30).copy()
    
    # Calculate how many predicted candles to show
    num_predicted = len(predicted_prices) if predicted_prices else 4
    
    # Get confidence bands from ML result
    confidence_bands = ml_result.get('confidence_bands', []) if ml_result else []
    mean_reversion = ml_result.get('mean_reversion', False) if ml_result else False
    
    # Create future candle data
    last_date = chart_df.index[-1]
    last_close = chart_df['Close'].iloc[-1]
    last_open = chart_df['Open'].iloc[-1]
    volatility = (chart_df['High'].iloc[-1] - chart_df['Low'].iloc[-1]) / last_close
    
    # Generate predicted candles
    predicted_candles = []
    current_predicted_price = last_close
    
    for i in range(num_predicted):
        # Calculate predicted price
        if predicted_prices and i < len(predicted_prices):
            pred_close = predicted_prices[i]
        else:
            # Default: gradual movement based on direction
            if direction == 'UP':
                change = confidence / 1000 * (i + 1)
            elif direction == 'DOWN':
                change = -confidence / 1000 * (i + 1)
            else:
                change = 0
            pred_close = last_close * (1 + change)
        
        # Use confidence bands for more realistic OHLC
        if i < len(confidence_bands):
            upper = confidence_bands[i]['upper']
            lower = confidence_bands[i]['lower']
        else:
            # Default spread based on volatility
            spread = abs(pred_close - current_predicted_price)
            upper = pred_close + spread * 0.5
            lower = pred_close - spread * 0.5
        
        # Generate OHLC for predicted candle
        pred_open = current_predicted_price
        pred_change = (pred_close - pred_open) / pred_open if pred_open > 0 else 0
        pred_high = max(pred_open, pred_close, upper)
        pred_low = min(pred_open, pred_close, lower)
        
        # Next time interval
        time_delta = pd.Timedelta(minutes=15) if interval == '15m' else \
                    pd.Timedelta(hours=1) if interval == '1h' else \
                    pd.Timedelta(hours=4) if interval == '4h' else \
                    pd.Timedelta(hours=1)
        
        pred_time = last_date + (time_delta * (i + 1))
        
        predicted_candles.append({
            'time': pred_time,
            'open': round(pred_open, 2),
            'high': round(pred_high, 2),
            'low': round(pred_low, 2),
            'close': round(pred_close, 2)
        })
        
        current_predicted_price = pred_close
    
    # Create figure
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{symbol} - AI Price Prediction ({duration_label})', 'Volume')
    )
    
    # ============ HISTORICAL CANDLES ============
    fig.add_trace(go.Candlestick(
        x=chart_df.index,
        open=chart_df['Open'],
        high=chart_df['High'],
        low=chart_df['Low'],
        close=chart_df['Close'],
        name='Historical',
        increasing_line_color='#00cc66',
        decreasing_line_color='#ff4b4b'
    ), row=1, col=1)
    
    # SMA 20
    if 'SMA_20' in chart_df.columns:
        fig.add_trace(go.Scatter(
            x=chart_df.index, y=chart_df['SMA_20'],
            name='SMA 20',
            line=dict(color='#00cc66', width=1.5)
        ), row=1, col=1)
    
    # SMA 50
    if 'SMA_50' in chart_df.columns:
        fig.add_trace(go.Scatter(
            x=chart_df.index, y=chart_df['SMA_50'],
            name='SMA 50',
            line=dict(color='#ff6b6b', width=1.5)
        ), row=1, col=1)
    
    # ============ PREDICTED FUTURE CANDLES ============
    if predicted_candles:
        pred_times = [c['time'] for c in predicted_candles]
        pred_opens = [c['open'] for c in predicted_candles]
        pred_highs = [c['high'] for c in predicted_candles]
        pred_lows = [c['low'] for c in predicted_candles]
        pred_closes = [c['close'] for c in predicted_candles]
        
        # Predicted candles with dashed outline styling
        fig.add_trace(go.Candlestick(
            x=pred_times,
            open=pred_opens,
            high=pred_highs,
            low=pred_lows,
            name='Predicted',
            increasing_line_color='#00cc66',
            decreasing_line_color='#ff4b4b',
            line=dict(width=2)
        ), row=1, col=1)
        
        # Predicted price line (connection)
        fig.add_trace(go.Scatter(
            x=[chart_df.index[-1]] + pred_times,
            y=[chart_df['Close'].iloc[-1]] + pred_closes,
            name='Prediction Path',
            mode='lines+markers',
            line=dict(color='#00cc66', width=2, dash='dot'),
            marker=dict(size=8, symbol='diamond', color='#00cc66')
        ), row=1, col=1)
        
        # ============ DYNAMIC CONFIDENCE BAND (FUNNEL) ============
        if confidence_bands:
            # Upper band
            upper_band = [b['upper'] for b in confidence_bands]
            lower_band = [b['lower'] for b in confidence_bands]
            
            # Extended for fill
            upper_full = [chart_df['Close'].iloc[-1]] + upper_band
            lower_full = [chart_df['Close'].iloc[-1]] + lower_band
            
            fig.add_trace(go.Scatter(
                x=pred_times + pred_times[::-1],
                y=upper_full + lower_full[::-1],
                fill='toself',
                fillcolor='rgba(0, 204, 102, 0.15)',
                line=dict(color='rgba(0, 0, 0, 0)'),
                name='Confidence Band (95%)',
                showlegend=True
            ), row=1, col=1)
        else:
            # Fallback: simple bands with volatility scaling
            upper_band = [p * (1 + volatility * (1 + i * 0.3)) for i, p in enumerate(pred_closes)]
            lower_band = [p * (1 - volatility * (1 + i * 0.3)) for i, p in enumerate(pred_closes)]
            
            fig.add_trace(go.Scatter(
                x=pred_times + pred_times[::-1],
                y=[chart_df['Close'].iloc[-1]] + upper_band + lower_band[::-1] + [chart_df['Close'].iloc[-1]],
                fill='toself',
                fillcolor='rgba(0, 204, 102, 0.15)',
                line=dict(color='rgba(0, 0, 0, 0)'),
                name='Confidence Band',
                showlegend=True
            ), row=1, col=1)
        
        # Future zone background
        fig.add_vrect(
            x0=chart_df.index[-1],
            x1=pred_times[-1],
            fillcolor='#00cc66',
            opacity=0.08,
            line_width=0,
            annotation_text='FUTURE PREDICTION',
            annotation_position='top left',
            row=1, col=1
        )
        
        # Mean reversion indicator
        if mean_reversion:
            fig.add_annotation(
                x=pred_times[len(pred_times)//2],
                y=np.mean(upper_band),
                text="⚠️ Mean Reversion Active",
                showarrow=True,
                arrowhead=2,
                arrowcolor='#ffa500',
                font=dict(color='#ffa500', size=12)
            )
    
    # ============ HISTORICAL CANDLES ============
    fig.add_trace(go.Candlestick(
        x=chart_df.index,
        open=chart_df['Open'],
        high=chart_df['High'],
        low=chart_df['Low'],
        close=chart_df['Close'],
        name='Historical',
        increasing_line_color='#00cc66',
        decreasing_line_color='#ff4b4b'
    ), row=1, col=1)
    
    # SMA 20
    if 'SMA_20' in chart_df.columns:
        fig.add_trace(go.Scatter(
            x=chart_df.index, y=chart_df['SMA_20'],
            name='SMA 20',
            line=dict(color='#00cc66', width=1.5)
        ), row=1, col=1)
    
    # SMA 50
    if 'SMA_50' in chart_df.columns:
        fig.add_trace(go.Scatter(
            x=chart_df.index, y=chart_df['SMA_50'],
            name='SMA 50',
            line=dict(color='#ff6b6b', width=1.5)
        ), row=1, col=1)
    
    # ============ PREDICTED FUTURE CANDLES ============
    if predicted_candles:
        pred_times = [c['time'] for c in predicted_candles]
        pred_opens = [c['open'] for c in predicted_candles]
        pred_highs = [c['high'] for c in predicted_candles]
        pred_lows = [c['low'] for c in predicted_candles]
        pred_closes = [c['close'] for c in predicted_candles]
        
        # Predicted candles with dashed outline styling
        fig.add_trace(go.Candlestick(
            x=pred_times,
            open=pred_opens,
            high=pred_highs,
            low=pred_lows,
            close=pred_closes,
            name='Predicted',
            increasing_line_color='#00cc66',
            decreasing_line_color='#ff4b4b',
            line=dict(width=2)
        ), row=1, col=1)
        
        # Predicted price line (connection)
        fig.add_trace(go.Scatter(
            x=[chart_df.index[-1]] + pred_times,
            y=[chart_df['Close'].iloc[-1]] + pred_closes,
            name='Prediction Path',
            mode='lines+markers',
            line=dict(color='#00cc66', width=2, dash='dot'),
            marker=dict(size=8, symbol='diamond', color='#00cc66')
        ), row=1, col=1)
        
        # Confidence band (upper/lower bounds)
        upper_band = [p * 1.02 for p in pred_closes]
        lower_band = [p * 0.98 for p in pred_closes]
        
        fig.add_trace(go.Scatter(
            x=pred_times + pred_times[::-1],
            y=upper_band + lower_band[::-1],
            fill='toself',
            fillcolor='rgba(0, 204, 102, 0.1)',
            line=dict(color='rgba(0, 0, 0, 0)'),
            name='Confidence Band',
            showlegend=True
        ), row=1, col=1)
        
        # Future zone background
        fig.add_vrect(
            x0=chart_df.index[-1],
            x1=pred_times[-1],
            fillcolor='#00cc66',
            opacity=0.08,
            line_width=0,
            annotation_text='FUTURE PREDICTION',
            annotation_position='top left',
            row=1, col=1
        )
    
    # ============ TRADE LEVELS ============
    # Entry Line
    fig.add_hline(
        y=levels['entry'],
        line_dash="solid",
        line_color="#00cc66",
        line_width=2,
        annotation_text="ENTRY",
        annotation_position="top right",
        row=1, col=1
    )
    
    # Stop Loss Line
    fig.add_hline(
        y=levels['sl'],
        line_dash="dot",
        line_color="#ff4b4b",
        line_width=2,
        annotation_text="SL",
        annotation_position="bottom right",
        row=1, col=1
    )
    
    # Take Profit Line
    fig.add_hline(
        y=levels['tp'],
        line_dash="dot",
        line_color="#00cc66",
        line_width=2,
        annotation_text="TP",
        annotation_position="top right",
        row=1, col=1
    )
    
    # Direction indicator
    if direction == 'UP':
        arrow_color = '#00cc66'
        arrow_text = '▲ BULLISH'
    elif direction == 'DOWN':
        arrow_color = '#ff4b4b'
        arrow_text = '▼ BEARISH'
    else:
        arrow_color = '#ffa500'
        arrow_text = '▶ NEUTRAL'
    
    # ============ RSI ============
    if 'RSI' in chart_df.columns:
        fig.add_trace(go.Scatter(
            x=chart_df.index, y=chart_df['RSI'],
            name='RSI',
            line=dict(color='#ffa500', width=2)
        ), row=2, col=1)
        
        fig.add_hline(y=70, line_dash="dash", line_color="#ff4b4b", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#00cc66", row=2, col=1)
    
    # ============ VOLUME ============
    if 'Volume' in chart_df.columns:
        colors = ['#00cc66' if chart_df['Close'].iloc[i] >= chart_df['Open'].iloc[i] 
                  else '#ff4b4b' for i in range(len(chart_df))]
        
        fig.add_trace(go.Bar(
            x=chart_df.index,
            y=chart_df['Volume'],
            name='Volume',
            marker_color=colors
        ), row=2, col=1)
    
    # ============ LAYOUT ============
    fig.update_layout(
        template='plotly_dark',
        height=700,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',
        font=dict(family="Roboto, sans-serif", size=12),
        margin=dict(t=80, b=40, l=40, r=40)
    )
    
    return fig


def main():
    # Sidebar
    st.sidebar.title("📈 AI Prediction")
    
    # Stock selection
    selected_stock = st.sidebar.selectbox(
        "Select Stock",
        ALL_STOCKS,
        index=0
    )
    
    # Timeframe selection
    timeframe = st.sidebar.selectbox(
        "Base Timeframe",
        ['15m', '1h', '4h'],
        index=1
    )
    
    # Duration selection (NEW!)
    st.sidebar.markdown("### ⏱️ Prediction Duration")
    
    duration_options = ['15 Minutes', '30 Minutes', '1 Hour', '2 Hours', '4 Hours', '8 Hours', '1 Day', '2 Days']
    selected_duration = st.sidebar.radio(
        "How far to predict?",
        duration_options,
        index=4  # Default: 4 Hours
    )
    
    # Map selection to steps
    duration_map = {
        '15 Minutes': ('15m', 1),
        '30 Minutes': ('30m', 2),
        '1 Hour': ('1h', 4),
        '2 Hours': ('1h', 8),
        '4 Hours': ('1h', 16),
        '8 Hours': ('1h', 32),
        '1 Day': ('1h', 48),
        '2 Days': ('1h', 96)
    }
    
    duration_key, num_candles = duration_map[selected_duration]
    
    # Generate button
    generate_btn = st.sidebar.button(
        "🔮 Generate Prediction",
        use_container_width=True
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 How it works")
    st.sidebar.markdown("""
    1. **Select Stock** - Choose any market
    2. **Select Duration** - How far to predict
    3. **Generate** - Click for AI prediction
    4. **View Chart** - See predicted future candles
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.caption("⚠️ For educational purposes only.")
    
    # Main content
    st.title("📈 AI Prediction Engine")
    st.markdown("---")
    
    if not generate_btn:
        st.markdown("""
        ### 👋 Welcome to AI Prediction!
        
        **How to use:**
        1. Select a stock from the sidebar
        2. Choose prediction duration (15m to 24h)
        3. Click **🔮 Generate Prediction**
        
        **What you'll get:**
        - 📊 Candlestick chart with predicted future candles
        - 🎯 Entry, SL, TP levels
        - 🤖 ML Confidence percentage
        - 📈 Professional visualization
        """)
        
        st.info("👈 Select a stock and click Generate Prediction to start!")
        return
    
    # Generate prediction
    with st.spinner(f"Analyzing {selected_stock} for {selected_duration}..."):
        # Get data
        period_map = {'15m': '5d', '1h': '30d', '4h': '60d'}
        df = get_data(selected_stock, timeframe, period_map.get(timeframe, '30d'))
        
        if df is None or len(df) < 20:
            st.error(f"Not enough data for {selected_stock}")
            return
        
        # Calculate indicators
        df = calculate_indicators(df)
        
        if df is None:
            st.error("Could not calculate indicators")
            return
        
        # Get current price
        current_price = df['Close'].iloc[-1]
        
        # Get ML prediction with future prices (use num_candles for prediction steps)
        ml_result = get_ml_prediction(df, num_steps=num_candles)
        
        # Get technical signal
        tech_result = get_technical_signal(df)
        
        # Combine signals
        ml_direction = ml_result.get('direction', 'HOLD')
        ml_confidence = ml_result.get('confidence', 50)
        future_prices = ml_result.get('future_prices', [])[:num_candles]
        
        tech_direction = tech_result.get('signal', 'HOLD')
        
        # Weighted combination
        if ml_direction == tech_direction:
            direction = ml_direction
            ml_confidence = min(ml_confidence + 15, 95)
        elif ml_direction == 'HOLD':
            direction = tech_direction
            ml_confidence = ml_confidence * 0.7
        elif tech_direction == 'HOLD':
            direction = ml_direction
        else:
            direction = ml_direction
            ml_confidence = ml_confidence * 0.6
        
        # Calculate levels
        levels = calculate_levels(current_price, future_prices, direction, ml_confidence)
        
        # Create chart with predicted candles
        fig = create_prediction_chart(
            df, selected_stock, current_price, levels, 
            direction, ml_confidence, timeframe, future_prices,
            selected_duration, ml_result
        )
        
        # ========================
        # Display Results
        # ========================
        
        # Header
        st.markdown(f"### 📊 {selected_stock} - {selected_duration} Prediction")
        
        # Direction Box
        if direction == 'UP':
            direction_display = "🟢 BUY / LONG"
            direction_color = "#00cc66"
        elif direction == 'DOWN':
            direction_display = "🔴 SELL / SHORT"
            direction_color = "#ff4b4b"
        else:
            direction_display = "⏸️ HOLD"
            direction_color = "#ffa500"
        
        # Confidence styling
        conf_class = "confidence-high" if ml_confidence >= 70 else "confidence-low"
        
        # Main signal display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style="background: {direction_color}20; border: 2px solid {direction_color}; padding: 20px; border-radius: 15px; text-align: center;">
                <div style="font-size: 24px; font-weight: bold; color: {direction_color};">
                    {direction_display}
                </div>
                <div style="font-size: 20px; font-weight: bold; color: {direction_color};">
                    {ml_confidence:.0f}% Confidence
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric("Current Price", f"${current_price:.2f}")
        
        with col3:
            st.metric("Trend", tech_result.get('trend', 'N/A'))
        
        with col4:
            st.metric("RSI", f"{tech_result.get('rsi', 50):.1f}")
        
        st.markdown("---")
        
        # Chart
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Could not generate chart")
        
        st.markdown("---")
        
        # Trade Levels
        st.markdown("### 🎯 Trade Levels")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style="border: 2px solid #00cc66; padding: 15px; border-radius: 10px; text-align: center;">
                <div style="color: #888; font-size: 14px;">ENTRY</div>
                <div style="color: #00cc66; font-size: 28px; font-weight: bold;">${levels['entry']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="border: 2px solid #ff4b4b; padding: 15px; border-radius: 10px; text-align: center;">
                <div style="color: #888; font-size: 14px;">STOP LOSS</div>
                <div style="color: #ff4b4b; font-size: 28px; font-weight: bold;">${levels['sl']:.2f}</div>
                <div style="color: #ff4b4b; font-size: 14px;">-{levels['sl_pct']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="border: 2px solid #00cc66; padding: 15px; border-radius: 10px; text-align: center;">
                <div style="color: #888; font-size: 14px;">TAKE PROFIT</div>
                <div style="color: #00cc66; font-size: 28px; font-weight: bold;">${levels['tp']:.2f}</div>
                <div style="color: #00cc66; font-size: 14px;">+{levels['tp_pct']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div style="background: #1a1a2e; padding: 15px; border-radius: 10px; text-align: center;">
                <div style="color: #888; font-size: 14px;">RISK/REWARD</div>
                <div style="color: #00cc66; font-size: 28px; font-weight: bold;">{levels['risk_reward']}:1</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Predicted prices table
        if future_prices:
            st.markdown("---")
            st.markdown("### 📈 Predicted Future Prices")
            
            pred_data = []
            for i, price in enumerate(future_prices[:8]):
                time_label = f"+{(i+1)*15}m" if timeframe == '15m' else f"+{i+1}h"
                pred_data.append({
                    'Step': time_label,
                    'Predicted Price': f"${price:.2f}",
                    'Change': f"{((price/current_price)-1)*100:+.2f}%"
                })
            
            st.dataframe(pd.DataFrame(pred_data), use_container_width=True)
        
        st.markdown("---")
        st.caption("⚠️ **Disclaimer**: This is for educational purposes only. Not financial advice.")


def show_prediction_page():
    main()


if __name__ == "__main__":
    main()
