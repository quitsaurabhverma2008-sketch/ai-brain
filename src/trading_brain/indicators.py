"""
Technical Indicators Module
============================
RSI, MACD, SMA, Bollinger Bands calculations
"""

import pandas as pd
import numpy as np
from typing import Optional


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index.
    
    Args:
        prices: Price series (usually Close)
        period: RSI period (default 14)
    
    Returns:
        Series with RSI values
    """
    delta = prices.diff()
    
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, 
                   signal: int = 9) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        prices: Price series (usually Close)
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)
    
    Returns:
        DataFrame with MACD, Signal, and Histogram
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    
    return pd.DataFrame({
        'MACD': macd,
        'Signal': signal_line,
        'Histogram': histogram
    })


def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return prices.rolling(window=period, min_periods=period).mean()


def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return prices.ewm(span=period, adjust=False).mean()


def calculate_bollinger_bands(prices: pd.Series, period: int = 20, 
                            std_dev: float = 2.0) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.
    
    Args:
        prices: Price series (usually Close)
        period: Moving average period (default 20)
        std_dev: Standard deviation multiplier (default 2.0)
    
    Returns:
        DataFrame with Upper, Middle, and Lower bands
    """
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    
    return pd.DataFrame({
        'BB_Upper': upper,
        'BB_Middle': middle,
        'BB_Lower': lower
    })


def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """
    Calculate Stochastic Oscillator.
    
    Args:
        df: DataFrame with High, Low, Close
        k_period: %K period (default 14)
        d_period: %D period (default 3)
    
    Returns:
        DataFrame with %K and %D
    """
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    
    k = 100 * (df['Close'] - low_min) / (high_max - low_min)
    d = k.rolling(window=d_period).mean()
    
    return pd.DataFrame({
        'Stoch_K': k,
        'Stoch_D': d
    })


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate Average Directional Index (ADX).
    
    Args:
        df: DataFrame with High, Low, Close
        period: ADX period (default 14)
    
    Returns:
        DataFrame with ADX, +DI, -DI
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(window=period).mean()
    
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return pd.DataFrame({
        'ADX': adx,
        'Plus_DI': plus_di,
        'Minus_DI': minus_di
    })


def calculate_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate Commodity Channel Index.
    
    Args:
        df: DataFrame with High, Low, Close
        period: CCI period (default 20)
    
    Returns:
        Series with CCI values
    """
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
    
    cci = (tp - sma_tp) / (0.015 * mad)
    
    return cci


def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """
    Calculate On Balance Volume.
    
    Args:
        df: DataFrame with Close, Volume
    
    Returns:
        Series with OBV
    """
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    
    return pd.Series(obv, index=df.index)


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range.
    
    Args:
        df: DataFrame with High, Low, Close
        period: ATR period (default 14)
    
    Returns:
        Series with ATR values
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(window=period).mean()
    
    return atr


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Volume Weighted Average Price.
    
    Args:
        df: DataFrame with High, Low, Close, Volume
    
    Returns:
        Series with VWAP
    """
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    cumsum_tp_vol = (tp * df['Volume']).cumsum()
    cumsum_vol = df['Volume'].cumsum()
    
    vwap = cumsum_tp_vol / cumsum_vol
    
    return vwap


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all technical indicators for a dataframe.
    
    Args:
        df: DataFrame with OHLC data (needs Close at minimum)
    
    Returns:
        DataFrame with all indicators added
    """
    df = df.copy()
    
    # RSI
    df['RSI'] = calculate_rsi(df['Close'], 14)
    
    # MACD
    macd_data = calculate_macd(df['Close'])
    df['MACD'] = macd_data['MACD']
    df['MACD_Signal'] = macd_data['Signal']
    df['MACD_Hist'] = macd_data['Histogram']
    
    # Moving Averages
    df['SMA_20'] = calculate_sma(df['Close'], 20)
    df['SMA_50'] = calculate_sma(df['Close'], 50)
    df['EMA_20'] = calculate_ema(df['Close'], 20)
    
    # Bollinger Bands
    bb_data = calculate_bollinger_bands(df['Close'])
    df['BB_Upper'] = bb_data['BB_Upper']
    df['BB_Middle'] = bb_data['BB_Middle']
    df['BB_Lower'] = bb_data['BB_Lower']
    
    # Stochastic Oscillator
    stoch_data = calculate_stochastic(df)
    df['Stoch_K'] = stoch_data['Stoch_K']
    df['Stoch_D'] = stoch_data['Stoch_D']
    
    # ADX
    adx_data = calculate_adx(df)
    df['ADX'] = adx_data['ADX']
    df['Plus_DI'] = adx_data['Plus_DI']
    df['Minus_DI'] = adx_data['Minus_DI']
    
    # CCI
    df['CCI'] = calculate_cci(df)
    
    # OBV
    df['OBV'] = calculate_obv(df)
    
    # ATR
    df['ATR'] = calculate_atr(df)
    
    # VWAP
    df['VWAP'] = calculate_vwap(df)
    
    # Additional
    df['Price_Change'] = df['Close'].pct_change() * 100
    df['Volume_SMA'] = df['Volume'].rolling(20).mean()
    
    return df


def get_latest_indicators(df: pd.DataFrame) -> dict:
    """
    Get latest values of all indicators.
    
    Args:
        df: DataFrame with indicators calculated
    
    Returns:
        Dictionary with latest indicator values
    """
    if df.empty or len(df) < 2:
        return {}
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    return {
        'close': latest.get('Close', 0),
        'rsi': latest.get('RSI', 50),
        'macd': latest.get('MACD', 0),
        'macd_signal': latest.get('MACD_Signal', 0),
        'macd_hist': latest.get('MACD_Hist', 0),
        'sma_20': latest.get('SMA_20', 0),
        'sma_50': latest.get('SMA_50', 0),
        'bb_upper': latest.get('BB_Upper', 0),
        'bb_middle': latest.get('BB_Middle', 0),
        'bb_lower': latest.get('BB_Lower', 0),
        'stoch_k': latest.get('Stoch_K', 50),
        'stoch_d': latest.get('Stoch_D', 50),
        'adx': latest.get('ADX', 0),
        'plus_di': latest.get('Plus_DI', 0),
        'minus_di': latest.get('Minus_DI', 0),
        'cci': latest.get('CCI', 0),
        'obv': latest.get('OBV', 0),
        'atr': latest.get('ATR', 0),
        'vwap': latest.get('VWAP', 0),
        'volume': latest.get('Volume', 0),
        'prev_close': prev.get('Close', 0),
    }


if __name__ == "__main__":
    # Test indicators
    from trading_brain.data_feed import DataFeed
    
    feed = DataFeed()
    df = feed.get_intraday_data("GOOGL", "1h", "5d")
    
    df = calculate_all_indicators(df)
    print(f"Indicators calculated. Rows: {len(df)}")
    print(df[['Close', 'RSI', 'MACD', 'SMA_20', 'BB_Upper']].tail(3))
