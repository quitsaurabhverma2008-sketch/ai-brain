"""
Step 5: Multi-Timeframe Backtesting
=================================
Test different timeframes and assets
"""

import pandas as pd
import numpy as np
import yfinance as yf


class MultiTimeframeBacktest:
    def __init__(self, capital: float = 10000):
        self.initial_capital = capital
    
    def download_data(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        print(f"\n[DOWNLOADING] {symbol} - {period} {interval}")
        stock = yf.Ticker(symbol)
        df = stock.history(period=period, interval=interval)
        
        if df.empty:
            print(f"  ERROR: No data for {symbol}")
            return None
        
        print(f"  Downloaded {len(df)} rows")
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
    
    def run_backtest(self, df: pd.DataFrame, sl_pct: float = 0.02, tp_pct: float = 0.06) -> dict:
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
                if prev['MACD'] <= prev['MACD_Signal'] and latest['MACD'] > latest['MACD_Signal']:
                    buy_score += 2
                if prev['MACD'] >= prev['MACD_Signal'] and latest['MACD'] < latest['MACD_Signal']:
                    sell_score += 2
                
                if latest['Close'] > latest['SMA_20']: buy_score += 1
                if latest['Close'] < latest['SMA_20']: sell_score += 1
                
                if buy_score >= 4 and trend == 'UP':
                    entry = latest['Close']
                    position = {
                        'type': 'LONG',
                        'entry': entry,
                        'sl': entry * (1 - sl_pct),
                        'tp': entry * (1 + tp_pct),
                        'size': capital / entry
                    }
                elif sell_score >= 4 and trend == 'DOWN':
                    entry = latest['Close']
                    position = {
                        'type': 'SHORT',
                        'entry': entry,
                        'sl': entry * (1 + sl_pct),
                        'tp': entry * (1 - tp_pct),
                        'size': capital / entry
                    }
            else:
                price = latest['Close']
                p = position
                exit_reason = None
                
                if p['type'] == 'LONG':
                    if price <= p['sl']:
                        exit_reason = 'SL'
                    elif price >= p['tp']:
                        exit_reason = 'TP'
                else:
                    if price >= p['sl']:
                        exit_reason = 'SL'
                    elif price <= p['tp']:
                        exit_reason = 'TP'
                
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
            'total_pnl': sum([t['pnl'] for t in trades]),
            'return_pct': ((capital - self.initial_capital) / self.initial_capital) * 100,
            'final_capital': capital,
            'trade_list': trades
        }


def main():
    print("\n" + "=" * 70)
    print("  MULTI-TIMEFRAME & ASSET TEST")
    print("=" * 70)
    
    backtest = MultiTimeframeBacktest(capital=10000)
    
    configs = [
        ("BTC-USD", "1y", "4h"),
        ("BTC-USD", "1y", "1d"),
        ("AAPL", "2y", "1d"),
        ("TSLA", "2y", "1d"),
        ("SPY", "2y", "1d"),
    ]
    
    results = []
    
    for symbol, period, interval in configs:
        df = backtest.download_data(symbol, period, interval)
        if df is None:
            continue
        
        df = backtest.calculate_indicators(df)
        if len(df) < 100:
            print(f"  Skipping {symbol} - not enough data")
            continue
        
        result = backtest.run_backtest(df)
        results.append({
            'symbol': symbol,
            'interval': interval,
            **result
        })
        
        print(f"\n  === {symbol} ({interval}) ===")
        print(f"  Trades: {result['trades']}, Win Rate: {result['win_rate']:.1f}%")
        print(f"  Return: {result['return_pct']:.2f}%, Final: ${result['final_capital']:,.0f}")
    
    print("\n" + "=" * 70)
    print("  SUMMARY - BEST TO WORST")
    print("=" * 70)
    
    sorted_results = sorted(results, key=lambda x: x['return_pct'], reverse=True)
    
    for i, r in enumerate(sorted_results, 1):
        status = "PROFIT!" if r['return_pct'] > 0 else "LOSS"
        print(f"{i}. {r['symbol']:<10} {r['interval']:<4} | "
              f"Trades: {r['trades']:>3} | "
              f"Win: {r['win_rate']:>5.1f}% | "
              f"Return: {r['return_pct']:>7.2f}% | {status}")
    
    return sorted_results


if __name__ == "__main__":
    main()