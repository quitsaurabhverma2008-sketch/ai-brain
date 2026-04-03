"""
Step 3: Technical Indicators (RSI, MACD, Moving Averages)
===========================================================
"""

import pandas as pd
import numpy as np


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators."""
    df = df.copy()
    
    # ===================
    # MOVING AVERAGES
    # ===================
    
    # Simple Moving Average (SMA)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Exponential Moving Average (EMA)
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # ===================
    # RSI (Relative Strength Index)
    # ===================
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # ===================
    # MACD (Moving Average Convergence Divergence)
    # ===================
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # ===================
    # BOLLINGER BANDS
    # ===================
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (2 * std)
    df['BB_Lower'] = df['BB_Middle'] - (2 * std)
    
    # ===================
    # ATR (Average True Range)
    # ===================
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    
    return df


def get_signal_from_indicators(df: pd.DataFrame) -> dict:
    """Generate trading signals based on indicators."""
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    
    signals = {
        "rsi": latest['RSI'],
        "macd": latest['MACD'],
        "macd_signal": latest['MACD_Signal'],
        "sma_20": latest['SMA_20'],
        "sma_50": latest['SMA_50'],
        "bb_position": "MIDDLE",
        "recommendation": "HOLD"
    }
    
    # RSI Signal
    rsi = latest['RSI']
    if rsi > 70:
        signals['rsi_signal'] = "OVERBOUGHT (Sell)"
    elif rsi < 30:
        signals['rsi_signal'] = "OVERSOLD (Buy)"
    else:
        signals['rsi_signal'] = "NEUTRAL"
    
    # MACD Signal (crossover)
    if latest['MACD'] > latest['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
        signals['macd_signal_text'] = "BULLISH CROSSOVER (Buy)"
    elif latest['MACD'] < latest['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
        signals['macd_signal_text'] = "BEARISH CROSSOVER (Sell)"
    else:
        signals['macd_signal_text'] = "NEUTRAL"
    
    # Price vs Moving Averages
    if latest['Close'] > latest['SMA_20'] > latest['SMA_50']:
        signals['trend'] = "BULLISH"
    elif latest['Close'] < latest['SMA_20'] < latest['SMA_50']:
        signals['trend'] = "BEARISH"
    else:
        signals['trend'] = "NEUTRAL"
    
    # Bollinger Band position
    if latest['Close'] > latest['BB_Upper']:
        signals['bb_position'] = "UPPER (Sell)"
    elif latest['Close'] < latest['BB_Lower']:
        signals['bb_position'] = "LOWER (Buy)"
    
    # Overall Recommendation
    buy_signals = 0
    sell_signals = 0
    
    if rsi < 30: buy_signals += 1
    if rsi > 70: sell_signals += 1
    if latest['MACD'] > latest['MACD_Signal']: buy_signals += 1
    else: sell_signals += 1
    if signals['trend'] == "BULLISH": buy_signals += 1
    elif signals['trend'] == "BEARISH": sell_signals += 1
    if signals['bb_position'] == "LOWER": buy_signals += 1
    elif signals['bb_position'] == "UPPER": sell_signals += 1
    
    if buy_signals > sell_signals + 1:
        signals['recommendation'] = "STRONG BUY"
    elif buy_signals > sell_signals:
        signals['recommendation'] = "BUY"
    elif sell_signals > buy_signals + 1:
        signals['recommendation'] = "STRONG SELL"
    elif sell_signals > buy_signals:
        signals['recommendation'] = "SELL"
    else:
        signals['recommendation'] = "HOLD"
    
    return signals


def main():
    """Main function for Step 3."""
    print("\n" + "=" * 60)
    print("  STEP 3: TECHNICAL INDICATORS")
    print("=" * 60)
    
    # Load data
    print("\n[1] Loading data...")
    df = pd.read_csv('data/BTC_USD.csv', parse_dates=True, index_col=0)
    print(f"    Loaded {len(df)} rows")
    
    # Calculate indicators
    print("\n[2] Calculating technical indicators...")
    df = calculate_indicators(df)
    print("    Indicators calculated: SMA, EMA, RSI, MACD, Bollinger Bands, ATR")
    
    # Get latest values
    latest = df.iloc[-1]
    print("\n" + "=" * 60)
    print("  LATEST INDICATOR VALUES")
    print("=" * 60)
    
    print(f"""
    +--------------------------------------------------------------+
    |  MOVING AVERAGES                                              |
    +--------------------------------------------------------------+
    |  SMA-20:     ${latest['SMA_20']:>10,.2f}  (20-period simple avg)      |
    |  SMA-50:     ${latest['SMA_50']:>10,.2f}  (50-period simple avg)      |
    |  EMA-12:     ${latest['EMA_12']:>10,.2f}  (12-period exp avg)         |
    |  EMA-26:     ${latest['EMA_26']:>10,.2f}  (26-period exp avg)         |
    +--------------------------------------------------------------+
    |  RSI:        {latest['RSI']:>10.2f}  (0-100, >70 overbought, <30 oversold)|
    +--------------------------------------------------------------+
    |  MACD                                                        |
    +--------------------------------------------------------------+
    |  MACD Line:  {latest['MACD']:>10.2f}                                     |
    |  Signal:     {latest['MACD_Signal']:>10.2f}                                     |
    |  Histogram:  {latest['MACD_Hist']:>10.2f}                                     |
    +--------------------------------------------------------------+
    |  BOLLINGER BANDS                                              |
    +--------------------------------------------------------------+
    |  Upper:      ${latest['BB_Upper']:>10,.2f}                                     |
    |  Middle:     ${latest['BB_Middle']:>10,.2f}                                     |
    |  Lower:      ${latest['BB_Lower']:>10,.2f}                                     |
    +--------------------------------------------------------------+
    |  ATR:        ${latest['ATR']:>10,.2f}  (Average True Range - volatility)   |
    +--------------------------------------------------------------+
    """)
    
    # Get signals
    print("\n[3] Analyzing signals...")
    signals = get_signal_from_indicators(df)
    
    print("\n" + "=" * 60)
    print("  TRADING SIGNALS ANALYSIS")
    print("=" * 60)
    
    print(f"""
    +--------------------------------------------------------------+
    |  Signal Type              Value                              |
    +--------------------------------------------------------------+
    |  RSI Signal:              {signals.get('rsi_signal', 'N/A'):<30}      |
    |  MACD Signal:             {signals.get('macd_signal_text', 'N/A'):<30}   |
    |  Trend (MA):              {signals.get('trend', 'N/A'):<30}         |
    |  Bollinger Position:      {signals.get('bb_position', 'N/A'):<30}      |
    +--------------------------------------------------------------+
    |                                                                  |
    |  FINAL RECOMMENDATION:  *** {signals['recommendation']} ***          |
    |                                                                  |
    +--------------------------------------------------------------+
    """)
    
    # Show some recent data with indicators
    print("\n[4] Recent data with indicators (last 5 rows):")
    print("-" * 60)
    display_cols = ['Close', 'SMA_20', 'RSI', 'MACD', 'MACD_Signal']
    print(df[display_cols].tail().to_string())
    
    # Save to file
    df.to_csv('data/BTC_USD_with_indicators.csv')
    print("\n[5] Data with indicators saved to: data/BTC_USD_with_indicators.csv")
    
    return df


if __name__ == "__main__":
    main()