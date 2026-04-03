"""
Step 5: Portfolio Optimization with Position Sizing
===================================================
"""

import pandas as pd
import numpy as np
import yfinance as yf


class PortfolioOptimizer:
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
    
    def download_data(self, symbol: str, period: str = "2y", interval: str = "1d"):
        stock = yf.Ticker(symbol)
        df = stock.history(period=period, interval=interval)
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
        
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2*std
        df['BB_Lower'] = df['BB_Middle'] - 2*std
        
        return df.dropna()
    
    def run_single_backtest(self, df: pd.DataFrame, sl_pct: float = 0.02, tp_pct: float = 0.06):
        capital = self.initial_capital
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
                    position = {'type': 'LONG', 'entry': entry, 'sl': entry*(1-sl_pct), 'tp': entry*(1+tp_pct), 'size': capital/entry}
                elif sell_score >= 4 and trend == 'DOWN':
                    entry = latest['Close']
                    position = {'type': 'SHORT', 'entry': entry, 'sl': entry*(1+sl_pct), 'tp': entry*(1-tp_pct), 'size': capital/entry}
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
                        pnl = (price - p['entry']) * p['size']
                    else:
                        pnl = (p['entry'] - price) * p['size']
                    trades.append({'pnl': pnl, 'type': p['type'], 'exit': exit_reason})
                    capital += pnl
                    position = None
        
        if position:
            price = df.iloc[-1]['Close']
            p = position
            if p['type'] == 'LONG':
                pnl = (price - p['entry']) * p['size']
            else:
                pnl = (p['entry'] - price) * p['size']
            trades.append({'pnl': pnl, 'type': p['type'], 'exit': 'END'})
            capital += pnl
        
        wins = len([t for t in trades if t['pnl'] > 0])
        
        return {
            'trades': len(trades),
            'wins': wins,
            'win_rate': (wins / len(trades) * 100) if trades else 0,
            'return_pct': ((capital - self.initial_capital) / self.initial_capital) * 100,
            'final_capital': capital,
            'trade_list': trades
        }
    
    def run_portfolio_backtest(self, stocks: list, weights: dict, parallel: bool = True) -> dict:
        """Run portfolio backtest with multiple stocks."""
        capital = self.initial_capital
        all_trades = []
        portfolio_value = [capital]
        
        # Get all stock data
        stock_data = {}
        for symbol in stocks:
            df = self.download_data(symbol, "2y", "1d")
            if df is not None and len(df) > 100:
                df = self.calculate_indicators(df)
                stock_data[symbol] = df
        
        # Find common date range
        if not stock_data:
            return {'error': 'No data'}
        
        common_dates = set(stock_data[stocks[0]].index)
        for symbol in stock_data:
            common_dates &= set(stock_data[symbol].index)
        common_dates = sorted(common_dates)
        
        # Run simulation
        positions = {s: None for s in stocks}
        
        for date_idx in range(50, len(common_dates)):
            current_date = common_dates[date_idx]
            
            # Check exits for all positions
            for symbol in stocks:
                if positions[symbol] is not None:
                    df = stock_data[symbol]
                    price = df.loc[current_date, 'Close']
                    p = positions[symbol]
                    exit_reason = None
                    
                    if p['type'] == 'LONG':
                        if price <= p['sl']: exit_reason = 'SL'
                        elif price >= p['tp']: exit_reason = 'TP'
                    else:
                        if price >= p['sl']: exit_reason = 'SL'
                        elif price <= p['tp']: exit_reason = 'TP'
                    
                    if exit_reason:
                        position_capital = capital * weights.get(symbol, 0)
                        if p['type'] == 'LONG':
                            pnl = (price - p['entry']) * (position_capital / p['entry'])
                        else:
                            pnl = (p['entry'] - price) * (position_capital / p['entry'])
                        all_trades.append({'symbol': symbol, 'pnl': pnl, 'type': p['type'], 'exit': exit_reason})
                        capital += pnl
            
            # Check entries for all stocks
            if parallel:
                for symbol in stocks:
                    if positions[symbol] is None and symbol in stock_data:
                        df = stock_data[symbol]
                        if current_date in df.index:
                            latest = df.loc[current_date]
                            prev = df.iloc[df.index.get_loc(current_date) - 1]
                            
                            buy_score = 0
                            trend = 'UP' if latest['SMA_20'] > latest['SMA_50'] else 'DOWN'
                            
                            if latest['RSI'] < 30: buy_score += 2
                            if latest['MACD'] > latest['MACD_Signal']: buy_score += 1
                            if prev['MACD'] <= prev['MACD_Signal'] and latest['MACD'] > latest['MACD_Signal']: buy_score += 2
                            if latest['Close'] > latest['SMA_20']: buy_score += 1
                            
                            if buy_score >= 4 and trend == 'UP':
                                entry = latest['Close']
                                position_capital = capital * weights.get(symbol, 0)
                                position_size = position_capital / entry
                                positions[symbol] = {
                                    'type': 'LONG', 'entry': entry, 'sl': entry * 0.98, 'tp': entry * 1.06, 'size': position_size
                                }
            
            portfolio_value.append(capital)
        
        # Calculate max drawdown
        peak = self.initial_capital
        max_dd = 0
        for val in portfolio_value:
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            if dd > max_dd:
                max_dd = dd
        
        wins = len([t for t in all_trades if t['pnl'] > 0])
        
        return {
            'total_trades': len(all_trades),
            'wins': wins,
            'win_rate': (wins / len(all_trades) * 100) if all_trades else 0,
            'return_pct': ((capital - self.initial_capital) / self.initial_capital) * 100,
            'final_capital': capital,
            'max_drawdown': max_dd * 100,
            'portfolio_value': portfolio_value
        }


def main():
    print("\n" + "=" * 70)
    print("  PORTFOLIO OPTIMIZATION WITH POSITION SIZING")
    print("=" * 70)
    
    optimizer = PortfolioOptimizer(initial_capital=100000)
    
    # Stock performance from previous test
    stock_performance = {
        'GOOGL': {'return': 49.21, 'win_rate': 53.3, 'rank': 1},
        'META': {'return': 36.10, 'win_rate': 53.8, 'rank': 2},
        'AAPL': {'return': 6.77, 'win_rate': 40.0, 'rank': 3},
        'SPY': {'return': 6.52, 'win_rate': 36.4, 'rank': 4},
        'MSFT': {'return': 1.61, 'win_rate': 30.8, 'rank': 5},
    }
    
    stocks = list(stock_performance.keys())
    
    # Define portfolios
    portfolios = {
        'CONSERVATIVE': {
            'GOOGL': 0.25,
            'META': 0.20,
            'AAPL': 0.20,
            'SPY': 0.20,
            'MSFT': 0.15,
        },
        'BALANCED': {
            'GOOGL': 0.30,
            'META': 0.25,
            'AAPL': 0.15,
            'SPY': 0.15,
            'MSFT': 0.15,
        },
        'AGGRESSIVE': {
            'GOOGL': 0.40,
            'META': 0.35,
            'AAPL': 0.10,
            'SPY': 0.10,
            'MSFT': 0.05,
        },
    }
    
    print("\n[Portfolio Allocations]")
    print("-" * 70)
    
    for name, weights in portfolios.items():
        print(f"\n{name}:")
        for symbol, weight in weights.items():
            print(f"  {symbol}: {weight*100:.0f}%")
    
    print("\n" + "=" * 70)
    print("  BACKTESTING ALL PORTFOLIOS")
    print("=" * 70)
    
    results = {}
    
    for name, weights in portfolios.items():
        print(f"\n[Running {name} Portfolio...]")
        
        result = optimizer.run_portfolio_backtest(stocks, weights)
        results[name] = result
        
        print(f"  Trades: {result['total_trades']}")
        print(f"  Win Rate: {result['win_rate']:.1f}%")
        print(f"  Return: {result['return_pct']:.2f}%")
        print(f"  Max Drawdown: {result['max_drawdown']:.2f}%")
    
    # Compare results
    print("\n" + "=" * 70)
    print("  PORTFOLIO COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Portfolio':<15} {'Return%':<12} {'Win Rate':<12} {'Max DD':<12} {'Trades':<10}")
    print("-" * 70)
    
    for name, result in results.items():
        print(f"{name:<15} {result['return_pct']:<12.2f} {result['win_rate']:<12.1f} {result['max_drawdown']:<12.2f} {result['total_trades']:<10}")
    
    # Risk-adjusted return (Sharpe-like)
    print("\n" + "=" * 70)
    print("  RISK-ADJUSTED RETURNS")
    print("=" * 70)
    
    for name, result in results.items():
        sharpe = result['return_pct'] / result['max_drawdown'] if result['max_drawdown'] > 0 else 0
        print(f"  {name}: Return/MaxDD = {sharpe:.2f}")
    
    # Best portfolio
    best = max(results.items(), key=lambda x: x[1]['return_pct'] / x[1]['max_drawdown'] if x[1]['max_drawdown'] > 0 else 0)
    
    print("\n" + "=" * 70)
    print("  RECOMMENDATION")
    print("=" * 70)
    
    print(f"""
    [BEST PORTFOLIO: {best[0]}]
    
    Return: {best[1]['return_pct']:.2f}%
    Win Rate: {best[1]['win_rate']:.1f}%
    Max Drawdown: {best[1]['max_drawdown']:.2f}%
    Risk-Adjusted Return: {best[1]['return_pct']/best[1]['max_drawdown']:.2f}
    
    Allocation:
""")
    
    for symbol, weight in portfolios[best[0]].items():
        print(f"  {symbol}: {weight*100:.0f}%")
    
    return results


if __name__ == "__main__":
    main()