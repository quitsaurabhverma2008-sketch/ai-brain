"""
Step 2: Data Exploration - Auto Calculations (No Plot)
========================================================
"""

import pandas as pd
import numpy as np


def load_data(filepath: str) -> pd.DataFrame:
    """Load data from CSV file."""
    df = pd.read_csv(filepath, parse_dates=True, index_col=0)
    return df


def explore_data(df: pd.DataFrame) -> dict:
    """Perform basic exploration of market data."""
    stats = {
        "total_rows": len(df),
        "date_range": (df.index[0], df.index[-1]),
        "current_price": df['Close'].iloc[-1],
        "highest_price": df['High'].max(),
        "lowest_price": df['Low'].min(),
        "average_price": df['Close'].mean(),
        "price_std": df['Close'].std(),
    }
    return stats


def calculate_returns_and_volatility(df: pd.DataFrame) -> tuple:
    """Calculate daily returns and volatility."""
    df = df.copy()
    
    # Daily returns: (current_close - previous_close) / previous_close
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Volatility = standard deviation of daily returns
    volatility = df['Daily_Return'].std()
    
    # Best and worst day returns
    best_day = df['Daily_Return'].max()
    worst_day = df['Daily_Return'].min()
    
    # Mean daily return
    mean_return = df['Daily_Return'].mean()
    
    # Cumulative returns
    df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod() - 1
    
    return df, {
        "volatility": volatility,
        "best_day": best_day,
        "worst_day": worst_day,
        "mean_return": mean_return
    }


def print_results(stats: dict, returns_info: dict, df: pd.DataFrame):
    """Print all results in structured format."""
    print("\n" + "=" * 60)
    print("        STEP 2: DATA EXPLORATION RESULTS")
    print("=" * 60)
    
    print("\n+-------------------------------------------------------------+")
    print("|  [DATASET OVERVIEW]                                        |")
    print("+-------------------------------------------------------------+")
    print(f"|  Total Data Points:     {stats['total_rows']:>35} |")
    print(f"|  Date Range:          {str(stats['date_range'][0])[:10]} to {str(stats['date_range'][1])[:10]:>20} |")
    print("+-------------------------------------------------------------+")
    
    print("\n+-------------------------------------------------------------+")
    print("|  [PRICE STATISTICS]                                        |")
    print("+-------------------------------------------------------------+")
    print(f"|  Current Price:       ${stats['current_price']:>33,.2f} |")
    print(f"|  Highest Price:       ${stats['highest_price']:>33,.2f} |")
    print(f"|  Lowest Price:        ${stats['lowest_price']:>33,.2f} |")
    print(f"|  Average Price:       ${stats['average_price']:>33,.2f} |")
    print(f"|  Price Std Dev:       ${stats['price_std']:>33,.2f} |")
    print("+-------------------------------------------------------------+")
    
    print("\n+-------------------------------------------------------------+")
    print("|  [RETURNS & VOLATILITY]                                     |")
    print("+-------------------------------------------------------------+")
    print(f"|  Mean Daily Return:   {returns_info['mean_return']*100:>33.3f}% |")
    print(f"|  Volatility (Std):    {returns_info['volatility']*100:>33.3f}% |")
    print("+-------------------------------------------------------------+")
    
    print("\n+-------------------------------------------------------------+")
    print("|  [BEST & WORST DAYS]                                        |")
    print("+-------------------------------------------------------------+")
    print(f"|  Best Day Return:     {returns_info['best_day']*100:>33.2f}% |")
    print(f"|  Worst Day Return:    {returns_info['worst_day']*100:>33.2f}% |")
    print("+-------------------------------------------------------------+")
    
    print("\n+-------------------------------------------------------------+")
    print("|  [FIRST 5 DAYS]                                             |")
    print("+-------------------------------------------------------------+")
    # Format the output manually
    print("                  Close        Daily_Return")
    for i in range(5):
        date = str(df.index[i])[:19]
        close = df['Close'].iloc[i]
        ret = df['Daily_Return'].iloc[i]
        print(f"|  {date}  {close:>10,.2f}  {ret:>12.4f}    |")
    print("+-------------------------------------------------------------+")
    
    print("\n+-------------------------------------------------------------+")
    print("|  [LAST 5 DAYS]                                              |")
    print("+-------------------------------------------------------------+")
    print("                  Close        Daily_Return")
    for i in range(-5, 0):
        date = str(df.index[i])[:19]
        close = df['Close'].iloc[i]
        ret = df['Daily_Return'].iloc[i]
        print(f"|  {date}  {close:>10,.2f}  {ret:>12.4f}    |")
    print("+-------------------------------------------------------------+")
    
    print("\n" + "=" * 60)
    print("        ALL CALCULATIONS COMPLETE!")
    print("=" * 60)


def main():
    """Main function to run Step 2."""
    print("\n" + "=" * 60)
    print("  STEP 2: DATA EXPLORATION - AUTO CALCULATIONS")
    print("=" * 60)
    
    # Load data from Step 1
    print("\n[1] Loading data from Step 1...")
    df = load_data('data/BTC_USD.csv')
    print(f"    Loaded {len(df)} rows of BTC-USD data")
    
    # Explore data
    print("\n[2] Calculating statistics...")
    stats = explore_data(df)
    
    # Calculate returns and volatility
    df, returns_info = calculate_returns_and_volatility(df)
    
    # Print all results
    print_results(stats, returns_info, df)
    
    return df


if __name__ == "__main__":
    main()