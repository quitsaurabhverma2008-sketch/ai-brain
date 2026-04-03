"""
Step 5: Portfolio Optimization (Fixed)
======================================
"""

import pandas as pd
import numpy as np
import yfinance as yf


class PortfolioOptimizer:
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
    
    def download_data(self, symbol: str):
        stock = yf.Ticker(symbol)
        df = stock.history(period="2y", interval="1d")
        return df
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        return df.dropna()
    
    def run_stock_backtest(self, df: pd.DataFrame, allocated_capital: float) -> dict:
        """Backtest single stock with allocated capital."""
        capital = allocated_capital
        trades = []
        position = None
        
        for i in range(1, len(df)):
            latest = df.iloc[i]
            prev = df.iloc[i-1]
            
            if position is None:
                buy_score = 0
                sell_score = 0
                trend = 'UP' if latest['SMA_20'] > latest['SMA_50'] else 'DOWN'
                
                if latest['RSI'] < 30: buy_score += 2
                if latest['RSI'] > 70: sell_score += 2
                if latest['MACD'] > latest['MACD_Signal']: buy_score += 1
                if latest['MACD'] < latest['MACD_Signal']: sell_score += 1
                if prev['MACD'] <= prev['MACD_Signal'] and latest['MACD'] > latest['MACD_Signal']: buy_score += 2
                if prev['MACD'] >= prev['MACD_Signal'] and latest['MACD'] < latest['MACD_Signal']: sell_score += 2
                if latest['Close'] > latest['SMA_20']: buy_score += 1
                if latest['Close'] < latest['SMA_20']: sell_score += 1
                
                if buy_score >= 4 and trend == 'UP':
                    entry = latest['Close']
                    position = {'type': 'LONG', 'entry': entry, 'sl': entry*0.98, 'tp': entry*1.06}
                elif sell_score >= 4 and trend == 'DOWN':
                    entry = latest['Close']
                    position = {'type': 'SHORT', 'entry': entry, 'sl': entry*1.02, 'tp': entry*0.94}
            else:
                price = latest['Close']
                p = position
                exit_reason = None
                
                if p['type'] == 'LONG':
                    if price <= p['sl']: exit_reason = 'SL'
                    elif price >= p['tp']: exit_reason = 'TP'
                else:
                    if price >= p['sl']: exit_reason = 'SL'
                    elif price <= p['tp']: exit_reason = 'TP'
                
                if exit_reason:
                    if p['type'] == 'LONG':
                        pnl = (price - p['entry']) * (capital / p['entry'])
                    else:
                        pnl = (p['entry'] - price) * (capital / p['entry'])
                    trades.append({'pnl': pnl})
                    capital += pnl
                    position = None
        
        if position:
            price = df.iloc[-1]['Close']
            p = position
            if p['type'] == 'LONG':
                pnl = (price - p['entry']) * (capital / p['entry'])
            else:
                pnl = (p['entry'] - price) * (capital / p['entry'])
            trades.append({'pnl': pnl})
            capital += pnl
        
        return {
            'initial': allocated_capital,
            'final': capital,
            'return_pct': ((capital - allocated_capital) / allocated_capital) * 100,
            'trades': len(trades)
        }
    
    def run_portfolio(self, stocks: dict) -> dict:
        """Run portfolio with separate capital allocation."""
        results = {}
        total_initial = sum(stocks.values())
        total_final = 0
        all_trades = 0
        
        for symbol, allocation in stocks.items():
            print(f"[{symbol}] Allocating ${allocation:,.0f} ({allocation/total_initial*100:.0f}%)...", end=" ")
            df = self.download_data(symbol)
            if df is None or len(df) < 100:
                print("FAILED")
                continue
            
            df = self.calculate_indicators(df)
            result = self.run_stock_backtest(df, allocation)
            results[symbol] = result
            total_final += result['final']
            all_trades += result['trades']
            print(f"Return: {result['return_pct']:.2f}%")
        
        total_return = ((total_final - total_initial) / total_initial) * 100
        
        return {
            'stocks': results,
            'total_initial': total_initial,
            'total_final': total_final,
            'total_return': total_return,
            'total_trades': all_trades
        }


def main():
    print("\n" + "=" * 70)
    print("  PORTFOLIO OPTIMIZATION WITH POSITION SIZING")
    print("=" * 70)
    
    optimizer = PortfolioOptimizer(initial_capital=100000)
    
    # Portfolio allocations
    portfolios = {
        'CONSERVATIVE': {
            'GOOGL': 25000,  # 25%
            'META': 20000,   # 20%
            'AAPL': 20000,   # 20%
            'SPY': 20000,    # 20%
            'MSFT': 15000,   # 15%
        },
        'BALANCED': {
            'GOOGL': 30000,  # 30%
            'META': 25000,   # 25%
            'AAPL': 15000,   # 15%
            'SPY': 15000,    # 15%
            'MSFT': 15000,   # 15%
        },
        'AGGRESSIVE': {
            'GOOGL': 40000,  # 40%
            'META': 35000,   # 35%
            'AAPL': 10000,   # 10%
            'SPY': 10000,    # 10%
            'MSFT': 5000,    # 5%
        },
    }
    
    results = {}
    
    print("\n" + "=" * 70)
    print("  BACKTESTING ALL PORTFOLIOS")
    print("=" * 70)
    
    for name, allocation in portfolios.items():
        print(f"\n[{name} Portfolio]")
        print("-" * 50)
        result = optimizer.run_portfolio(allocation)
        results[name] = result
    
    # Comparison
    print("\n" + "=" * 70)
    print("  PORTFOLIO COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Portfolio':<15} {'Initial':<15} {'Final':<15} {'Return%':<12} {'Trades'}")
    print("-" * 70)
    
    for name, result in results.items():
        print(f"{name:<15} ${result['total_initial']:>12,.0f} ${result['total_final']:>12,.0f} {result['total_return']:>10.2f}% {result['total_trades']}")
    
    # Stock breakdown
    print("\n" + "=" * 70)
    print("  STOCK BREAKDOWN BY PORTFOLIO")
    print("=" * 70)
    
    for name, result in results.items():
        print(f"\n{name}:")
        for symbol, stock_result in result['stocks'].items():
            print(f"  {symbol}: ${stock_result['initial']:,.0f} -> ${stock_result['final']:,.0f} ({stock_result['return_pct']:+.2f}%)")
    
    # Best portfolio
    best = max(results.items(), key=lambda x: x[1]['total_return'])
    
    print("\n" + "=" * 70)
    print("  RECOMMENDATION")
    print("=" * 70)
    
    print(f"""
    [BEST PORTFOLIO: {best[0]}]
    
    Total Return: {best[1]['total_return']:.2f}%
    Final Capital: ${best[1]['total_final']:,.2f}
    Total Trades: {best[1]['total_trades']}
    
    Allocation:
""")
    
    for symbol, allocation in portfolios[best[0]].items():
        print(f"  {symbol}: ${allocation:,} ({allocation/100000*100:.0f}%)")
    
    return results


if __name__ == "__main__":
    main()