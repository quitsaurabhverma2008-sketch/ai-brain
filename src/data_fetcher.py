"""
Step 1: Data Fetcher
====================
Download stock market data using yfinance
"""

import yfinance as yf
import pandas as pd
import os


def download_stock_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Download stock data from Yahoo Finance.
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        
        if df.empty:
            print(f"Warning: No data found for {ticker}")
            return None
            
        print(f"Downloaded {len(df)} data points for {ticker}")
        return df
        
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        return None


def save_to_csv(df: pd.DataFrame, ticker: str, folder: str = "data") -> str:
    """Save DataFrame to CSV file."""
    os.makedirs(folder, exist_ok=True)
    filename = f"{folder}/{ticker.replace('-', '_')}.csv"
    df.to_csv(filename)
    print(f"Saved to {filename}")
    return filename


def main():
    """Download BTC-USD data for Step 2."""
    print("=" * 50)
    print("Downloading data for Step 2...")
    print("=" * 50)
    
    # Download Bitcoin data (30 days, 1 hour intervals for more data points)
    df = download_stock_data("BTC-USD", period="30d", interval="1h")
    
    if df is not None:
        print(f"\nData preview:")
        print(df.head())
        
        # Save to data folder
        os.makedirs("data", exist_ok=True)
        filepath = save_to_csv(df, "BTC-USD", "data")
        print(f"\nData saved to: {filepath}")
        return df
    
    return None


if __name__ == "__main__":
    main()