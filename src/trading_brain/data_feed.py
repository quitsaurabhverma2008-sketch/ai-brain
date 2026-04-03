"""
Data Feed Module - Intraday Trading
===================================
Real-time stock data from Yahoo Finance
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional


class DataFeed:
    """Real-time stock data feed for intraday trading"""
    
    def __init__(self, cache_seconds: int = 60):
        self.cache_seconds = cache_seconds
        self.cache: Dict[str, pd.DataFrame] = {}
        self.last_refresh: Dict[str, datetime] = {}
    
    def get_intraday_data(self, symbol: str, interval: str = "1h", period: str = "5d") -> pd.DataFrame:
        """Get intraday data for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(interval=interval, period=period)
            
            if df.empty:
                return pd.DataFrame()
            
            df.index = pd.to_datetime(df.index)
            return df
            
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()
    
    def get_multiple_stocks(self, symbols: List[str], interval: str = "1h", 
                          period: str = "5d") -> Dict[str, pd.DataFrame]:
        """Get data for multiple stocks."""
        data = {}
        
        for symbol in symbols:
            df = self.get_intraday_data(symbol, interval, period)
            if not df.empty:
                data[symbol] = df
        
        return data
    
    def get_live_quote(self, symbol: str) -> Optional[dict]:
        """Get current quote for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            prev = info.get('previousClose', info.get('currentPrice', 0))
            
            return {
                'symbol': symbol,
                'price': info.get('currentPrice', 0),
                'open': info.get('open', 0),
                'high': info.get('dayHigh', 0),
                'low': info.get('dayLow', 0),
                'volume': info.get('volume', 0),
                'prev_close': prev,
                'change': info.get('currentPrice', 0) - prev,
                'change_pct': ((info.get('currentPrice', 0) - prev) / max(prev, 0.01)) * 100,
                'timestamp': datetime.now()
            }
        except:
            return None
    
    def get_quotes(self, symbols: List[str]) -> Dict[str, dict]:
        """Get quotes for multiple symbols."""
        return {s: self.get_live_quote(s) for s in symbols if self.get_live_quote(s)}


if __name__ == "__main__":
    feed = DataFeed()
    df = feed.get_intraday_data("GOOGL", "1h", "5d")
    print(f"GOOGL: {len(df)} candles")
    q = feed.get_live_quote("GOOGL")
    print(f"Price: ${q['price']}, Change: {q['change_pct']:.2f}%")
