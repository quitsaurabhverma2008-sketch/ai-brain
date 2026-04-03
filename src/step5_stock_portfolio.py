"""
Step 5: Stock Portfolio Backtesting
===================================
"""

import pandas as pd
import numpy as np
import yfinance as yf


class StockPortfolioBacktest:
    def __init__(self, capital: float = 10000):
        self.initial_capital = capital
    
    def download_data(self, symbol: str, period: str = "2y", interval: str = "1d"):
        print(f"[{symbol}] Downloading...", end=" ")
        stock = yf.Ticker(symbol)
        df = stock.history(period=period, interval=interval)
        
        if df.empty:
            print("FAILED")
            return None
        
        print(f"{len(df)} rows")
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
    
    def run_backtest(self, df: pd.DataFrame, sl_pct: float = 0.02, tp_pct: float = 0.06):
        capital = self.initial_capital
        trades = []
        position = None
        entry_date = None
        
        for i in range(1, len(df)):
            latest = df.iloc[i]
            prev = df.iloc[i-1]
            
            if entry_date is not None:
                hours_diff = (latest.name - entry_date).total_seconds() / 86400
                if hours_diff < 1 and position is not None:
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
                    continue
            
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
                    entry_date = latest.name
                elif sell_score >= 4 and trend == 'DOWN':
                    entry = latest['Close']
                    position = {'type': 'SHORT', 'entry': entry, 'sl': entry*(1+sl_pct), 'tp': entry*(1-tp_pct), 'size': capital/entry}
                    entry_date = latest.name
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
                    entry_date = None
        
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
            'total_pnl': sum([t['pnl'] for t in trades]),
            'return_pct': ((capital - self.initial_capital) / self.initial_capital) * 100,
            'final_capital': capital,
        }


def main():
    print("\n" + "=" * 70)
    print("  STOCK PORTFOLIO BACKTESTING")
    print("=" * 70)
    
    stocks = ["AAPL", "SPY", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSMC"]
    
    backtest = StockPortfolioBacktest(capital=10000)
    results = []
    
    print("\n[Running backtests on stocks...]")
    
    for symbol in stocks:
        df = backtest.download_data(symbol, "2y", "1d")
        if df is None:
            continue
        
        df = backtest.calculate_indicators(df)
        if len(df) < 100:
            continue
        
        result = backtest.run_backtest(df)
        results.append({'symbol': symbol, **result})
    
    results.sort(key=lambda x: x['return_pct'], reverse=True)
    
    print("\n" + "=" * 70)
    print("  STOCK RANKING - BY RETURN")
    print("=" * 70)
    
    print(f"\n{'Rank':<5} {'Stock':<8} {'Trades':<8} {'Win%':<8} {'Return%':<10} {'Final':<12} {'Status'}")
    print("-" * 70)
    
    for i, r in enumerate(results, 1):
        status = "PROFIT!" if r['return_pct'] > 0 else "LOSS"
        print(f"{i:<5} {r['symbol']:<8} {r['trades']:<8} {r['win_rate']:<8.1f} {r['return_pct']:<10.2f} ${r['final_capital']:<11,.0f} {status}")
    
    profitable = [r for r in results if r['return_pct'] > 0]
    
    print("\n" + "=" * 70)
    print("  TOP PERFORMING STOCKS")
    print("=" * 70)
    
    if profitable:
        top3 = profitable[:3]
        print(f"\n[TOP 3 STOCKS - RECOMMENDED FOR PORTFOLIO]")
        
        total_return = sum(r['return_pct'] for r in top3)
        avg_win = sum(r['win_rate'] for r in top3) / len(top3)
        
        for i, r in enumerate(top3, 1):
            print(f"\n{i}. {r['symbol']}")
            print(f"   Return: {r['return_pct']:.2f}%")
            print(f"   Win Rate: {r['win_rate']:.1f}%")
            print(f"   Trades: {r['trades']}")
            print(f"   Final Capital: ${r['final_capital']:,.2f}")
        
        print(f"\n[PORTFOLIO SUMMARY]")
        print(f"   Combined Return: {total_return:.2f}%")
        print(f"   Average Win Rate: {avg_win:.1f}%")
    
    return results


if __name__ == "__main__":
    main()