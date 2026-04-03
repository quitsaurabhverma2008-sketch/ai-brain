"""
Data Collector Module
=====================
Download historical data for 200+ markets
"""

import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


STOCK_LISTS = {
    'us_stocks': [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'JPM', 'JNJ',
        'V', 'PG', 'UNH', 'MA', 'HD', 'DIS', 'PYPL', 'BAC', 'ADBE', 'CRM',
        'NFLX', 'INTC', 'VZ', 'T', 'XOM', 'KO', 'PEP', 'WMT', 'ABT', 'MRK',
        'CVX', 'LLY', 'PFE', 'ABBV', 'TMO', 'COST', 'AVGO', 'NEE', 'DHR', 'NKE',
        'TXN', 'QCOM', 'HON', 'UPS', 'PM', 'MS', 'GS', 'BLK', 'IBM', 'AMD',
        'GE', 'CAT', 'BA', 'MMM', 'RTX', 'LOW', 'SPGI', 'LRCX', 'MU', 'SBUX',
        'SCHW', 'BKNG', 'ISRG', 'MDLZ', 'GILD', 'TJX', 'ADP', 'VRTX', 'ZTS', 'REGN',
        'SYK', 'PLD', 'BDX', 'BMY', 'CI', 'CB', 'MO', 'SO', 'DUK', 'CCI',
        'AON', 'AJG', 'MCK', 'BSX', 'EW', 'EL', 'GM', 'F', 'ORCL', 'CSCO',
        'INFY', 'TCS', 'WFC', 'C', 'USB', 'AXP', 'AIG', 'MET', 'PRU', 'AFL',
        'TGT', 'HD', 'LOW', 'COST', 'SBUX', 'MCD', 'CMG', 'YUM', 'DPZ', 'QRTEA'
    ],
    
    'indian_stocks': [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'SBIN.NS',
        'BAJFINANCE.NS', 'KOTAKBANK.NS', 'ITC.NS', 'HINDUNILVR.NS', 'LT.NS', 'AXISBANK.NS',
        'MARUTI.NS', 'SUNPHARMA.NS', 'TITAN.NS', 'WIPRO.NS', 'ONGC.NS', 'NTPC.NS',
        'POWERGRID.NS', 'BHARTIARTL.NS', 'ADANIENT.NS', 'ADANIPORTS.NS', 'CIPLA.NS',
        'DRREDDY.NS', 'GRASIM.NS', 'HCLTECH.NS', 'INDUSIND.NS', 'JSWSTEEL.NS',
        'KERNEL.NS', 'LTI.NS', 'M&M.NS', 'NESTLEIND.NS', 'RECLTD.NS', 'SHREECEM.NS',
        'VEDL.NS', 'VOLTAS.NS', 'YESBANK.NS', 'IDBI.NS', 'IDFCFIRSTB.NS', 'BANDHANBNK.NS',
        'FEDERALBNK.NS', 'CANBK.NS', 'PNB.NS', 'UNIONBANK.NS', 'IOB.NS', 'BPCL.NS',
        'HPCL.NS', 'IOC.NS', 'GAIL.NS'
    ],
    
    'crypto': [
        'BTC-USD', 'ETH-USD', 'USDT-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD', 
        'USDC-USD', 'ADA-USD', 'AVAX-USD', 'DOGE-USD', 'DOT-USD', 'TRX-USD',
        'LINK-USD', 'MATIC-USD', 'SHIB-USD', 'LTC-USD', 'DAI-USD', 'BCH-USD',
        'UNI-USD', 'ATOM-USD', 'ETC-USD', 'XLM-USD', 'XMR-USD', 'ALGO-USD',
        'VET-USD', 'FIL-USD', 'HBAR-USD', 'NEAR-USD', 'APT-USD', 'ARB-USD'
    ],
    
    'etfs': [
        'SPY', 'QQQ', 'IWM', 'DIA', 'VOO', 'VTI', 'VEA', 'VWO', 'VTV', 'VIG',
        'SCHD', 'JEPI', 'JEPQ', 'VNQ', 'GLD', 'SLV', 'TLT', 'HYG', 'LQD', 'EMB'
    ]
}


class DataCollector:
    """Collect historical data for multiple markets"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        self.all_symbols = (
            STOCK_LISTS['us_stocks'] + 
            STOCK_LISTS['indian_stocks'] + 
            STOCK_LISTS['crypto'] + 
            STOCK_LISTS['etfs']
        )
        
    def get_data_path(self, symbol: str) -> str:
        """Get file path for a symbol"""
        safe_name = symbol.replace('.', '_').replace('-', '_')
        return os.path.join(self.data_dir, f"{safe_name}.csv")
    
    def download_data(self, symbol: str, period: str = "20y", 
                     interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        Download data for a single symbol.
        
        Args:
            symbol: Stock/crypto symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 5m, 15m, 30m, 60m, 1d, 1wk, 1mo)
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval, auto_adjust=True)
            
            if df.empty:
                print(f"No data for {symbol}")
                return None
            
            df = df.dropna()
            df = df.reset_index()
            df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
            df.set_index('Datetime', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error downloading {symbol}: {e}")
            return None
    
    def download_intraday(self, symbol: str, interval: str = "1h", 
                         period: str = "730d") -> Optional[pd.DataFrame]:
        """
        Download intraday data for a symbol.
        
        Args:
            symbol: Stock/crypto symbol
            interval: Intraday interval (1m, 2m, 5m, 15m, 30m, 60m, 1h)
            period: Period to download (1d, 5d, 60d, 730d, etc.)
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(interval=interval, period=period, auto_adjust=True)
            
            if df.empty:
                return None
            
            df = df.dropna()
            df = df.reset_index()
            
            return df
            
        except Exception as e:
            return None
    
    def save_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """Save data to CSV"""
        if df is None or df.empty:
            return False
        
        try:
            path = self.get_data_path(symbol)
            df.to_csv(path, index=False)
            return True
        except Exception as e:
            print(f"Error saving {symbol}: {e}")
            return False
    
    def load_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load data from CSV"""
        path = self.get_data_path(symbol)
        
        if not os.path.exists(path):
            return None
        
        try:
            df = pd.read_csv(path)
            if 'Datetime' in df.columns:
                df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
                df.set_index('Datetime', inplace=True)
            elif 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], utc=True)
                df.set_index('Date', inplace=True)
            return df
        except Exception as e:
            print(f"Error loading {symbol}: {e}")
            return None
    
    def download_all(self, category: str = "all", 
                    period: str = "20y", interval: str = "1d",
                    save: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Download data for all symbols in a category.
        
        Args:
            category: us_stocks, indian_stocks, crypto, etfs, or all
            period: Data period
            interval: Data interval
            save: Whether to save to CSV
        
        Returns:
            Dictionary of symbol -> DataFrame
        """
        if category == "all":
            symbols = self.all_symbols
        else:
            symbols = STOCK_LISTS.get(category, [])
        
        results = {}
        total = len(symbols)
        
        print(f"\nDownloading {total} symbols from {category}...")
        
        for i, symbol in enumerate(symbols):
            print(f"[{i+1}/{total}] {symbol}...", end=" ")
            
            df = self.download_data(symbol, period, interval)
            
            if df is not None and len(df) > 0:
                if save:
                    self.save_data(df, symbol)
                results[symbol] = df
                print(f"OK ({len(df)} rows)")
            else:
                print("FAIL")
        
        print(f"\nDownload complete: {len(results)}/{total} symbols")
        
        return results
    
    def download_intraday_all(self, category: str = "all",
                              interval: str = "1h",
                              period: str = "730d",
                              save: bool = True) -> Dict[str, pd.DataFrame]:
        """Download intraday data for all symbols"""
        
        if category == "all":
            symbols = self.all_symbols
        else:
            symbols = STOCK_LISTS.get(category, [])
        
        results = {}
        total = len(symbols)
        
        print(f"\nDownloading intraday data for {total} symbols...")
        
        for i, symbol in enumerate(symbols):
            print(f"[{i+1}/{total}] {symbol}...", end=" ")
            
            df = self.download_intraday(symbol, interval, period)
            
            if df is not None and len(df) > 0:
                if save:
                    self.save_data(df, symbol)
                results[symbol] = df
                print(f"OK ({len(df)} rows)")
            else:
                print("FAIL")
        
        print(f"\nIntraday download complete: {len(results)}/{total}")
        
        return results
    
    def get_stats(self) -> Dict:
        """Get data collection stats"""
        stats = {
            'us_stocks': len(STOCK_LISTS['us_stocks']),
            'indian_stocks': len(STOCK_LISTS['indian_stocks']),
            'crypto': len(STOCK_LISTS['crypto']),
            'etfs': len(STOCK_LISTS['etfs']),
            'total': len(self.all_symbols)
        }
        
        saved_count = 0
        for symbol in self.all_symbols:
            path = self.get_data_path(symbol)
            if os.path.exists(path):
                saved_count += 1
        
        stats['downloaded'] = saved_count
        
        return stats
    
    def create_training_dataset(self, lookback: int = 24,
                                target: int = 1) -> pd.DataFrame:
        """
        Create training dataset from all collected data.
        
        Args:
            lookback: Number of hours to look back
            target: Hours ahead to predict
        
        Returns:
            DataFrame with features and target
        """
        from indicators import calculate_all_indicators
        
        all_data = []
        
        for symbol in self.all_symbols:
            df = self.load_data(symbol)
            
            if df is None or len(df) < lookback + target + 50:
                continue
            
            df = calculate_all_indicators(df)
            
            for i in range(lookback, len(df) - target):
                row = {
                    'symbol': symbol,
                    'target': 1 if df['Close'].iloc[i + target] > df['Close'].iloc[i] else 0
                }
                
                feature_cols = ['RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
                               'SMA_20', 'SMA_50', 'EMA_20',
                               'BB_Upper', 'BB_Middle', 'BB_Lower',
                               'Stoch_K', 'Stoch_D', 'ADX', 'Plus_DI', 'Minus_DI',
                               'CCI', 'ATR', 'VWAP', 'Volume']
                
                for col in feature_cols:
                    if col in df.columns:
                        row[col] = df[col].iloc[i]
                
                all_data.append(row)
        
        return pd.DataFrame(all_data)


if __name__ == "__main__":
    collector = DataCollector()
    
    stats = collector.get_stats()
    print(f"\nData Collection Stats:")
    print(f"  US Stocks: {stats['us_stocks']}")
    print(f"  Indian Stocks: {stats['indian_stocks']}")
    print(f"  Crypto: {stats['crypto']}")
    print(f"  ETFs: {stats['etfs']}")
    print(f"  Total: {stats['total']}")
    print(f"  Downloaded: {stats['downloaded']}")
    
    print("\n" + "="*50)
    print("Downloading data for all markets (this may take a while)...")
    print("="*50)
    
    collector.download_all(category="all", period="20y", interval="1d")
