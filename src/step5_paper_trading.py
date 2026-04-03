"""
Step 5: Paper Trading Simulation
================================
Simulate live trading with latest market data
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time


class PaperTrader:
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.daily_pnl = []
    
    def download_recent_data(self, symbol: str, days: int = 60):
        """Download recent market data."""
        stock = yf.Ticker(symbol)
        df = stock.history(period=f"{days}d", interval="1d")
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
    
    def check_signal(self, df: pd.DataFrame) -> str:
        """Check for buy/sell signal."""
        if len(df) < 2:
            return 'HOLD'
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        buy_score = 0
        sell_score = 0
        trend = 'UP' if latest['SMA_20'] > latest['SMA_50'] else 'DOWN'
        
        # RSI
        if latest['RSI'] < 30: buy_score += 2
        if latest['RSI'] > 70: sell_score += 2
        
        # MACD
        if latest['MACD'] > latest['MACD_Signal']: buy_score += 1
        if latest['MACD'] < latest['MACD_Signal']: sell_score += 1
        if prev['MACD'] <= prev['MACD_Signal'] and latest['MACD'] > latest['MACD_Signal']: buy_score += 2
        if prev['MACD'] >= prev['MACD_Signal'] and latest['MACD'] < latest['MACD_Signal']: sell_score += 2
        
        # Price vs MA
        if latest['Close'] > latest['SMA_20']: buy_score += 1
        if latest['Close'] < latest['SMA_20']: sell_score += 1
        
        # Signal
        if buy_score >= 4 and trend == 'UP':
            return 'BUY'
        elif sell_score >= 4 and trend == 'DOWN':
            return 'SELL'
        
        return 'HOLD'
    
    def execute_trade(self, symbol: str, action: str, price: float, date):
        """Execute a trade with slippage simulation."""
        slippage = 0.001  # 0.1% slippage
        execution_price = price * (1 + slippage) if action == 'BUY' else price * (1 - slippage)
        
        if action == 'BUY' and self.cash >= execution_price * 10:  # Min 10 shares
            shares = int(self.cash * 0.1 / execution_price)  # 10% of capital
            cost = shares * execution_price
            
            self.positions[symbol] = {
                'shares': shares,
                'entry_price': execution_price,
                'entry_date': date,
                'sl': execution_price * 0.98,
                'tp': execution_price * 1.06
            }
            self.cash -= cost
            
            self.trades.append({
                'symbol': symbol,
                'action': 'BUY',
                'shares': shares,
                'price': execution_price,
                'value': cost,
                'date': date,
                'pnl': 0,
                'status': 'OPEN'
            })
            
            print(f"  [BUY] {symbol}: {shares} shares @ ${execution_price:.2f}")
            
        elif action == 'SELL' and symbol in self.positions:
            pos = self.positions[symbol]
            proceeds = pos['shares'] * execution_price
            pnl = proceeds - (pos['shares'] * pos['entry_price'])
            
            self.trades.append({
                'symbol': symbol,
                'action': 'SELL',
                'shares': pos['shares'],
                'price': execution_price,
                'value': proceeds,
                'date': date,
                'pnl': pnl,
                'status': 'CLOSED',
                'entry_price': pos['entry_price'],
                'holding_days': (date - pos['entry_date']).days
            })
            
            self.cash += proceeds
            del self.positions[symbol]
            
            print(f"  [SELL] {symbol}: {pos['shares']} shares @ ${execution_price:.2f} | PnL: ${pnl:.2f}")
    
    def check_exits(self, df: pd.DataFrame, date):
        """Check if any positions should be exited."""
        symbols_to_close = []
        
        for symbol, pos in self.positions.items():
            if symbol not in df.index:
                continue
                
            current_price = df.loc[symbol, 'Close'] if symbol in df.columns else df.iloc[-1]['Close']
            
            # Check SL/TP
            if pos['type'] == 'LONG':
                if current_price <= pos['sl']:
                    symbols_to_close.append((symbol, 'STOP_LOSS', current_price))
                elif current_price >= pos['tp']:
                    symbols_to_close.append((symbol, 'TAKE_PROFIT', current_price))
            else:
                if current_price >= pos['sl']:
                    symbols_to_close.append((symbol, 'STOP_LOSS', current_price))
                elif current_price <= pos['tp']:
                    symbols_to_close.append((symbol, 'TAKE_PROFIT', current_price))
        
        return symbols_to_close
    
    def run_paper_trading(self, symbols: list, weeks: int = 4) -> dict:
        """Run paper trading simulation."""
        print("\n" + "=" * 70)
        print("  PAPER TRADING SIMULATION")
        print("=" * 70)
        
        print(f"\n[Configuration]")
        print(f"  Initial Capital: ${self.initial_capital:,.2f}")
        print(f"  Duration: {weeks} weeks")
        print(f"  Symbols: {', '.join(symbols)}")
        
        # Download more data for indicators
        print("\n[Downloading market data...]")
        stock_data = {}
        
        for symbol in symbols:
            print(f"  {symbol}...", end=" ")
            df = self.download_recent_data(symbol, days=180)  # 6 months
            if df is not None and len(df) > 50:
                df = self.calculate_indicators(df)
                stock_data[symbol] = df
                print(f"{len(df)} days")
        
        if not stock_data:
            print("ERROR: No data available")
            return {}
        
        # Run simulation day by day
        print("\n[Running simulation...]")
        
        min_data_len = min([len(df) for df in stock_data.values()])
        start_idx = min(20, min_data_len - 1)
        
        start_date = list(stock_data.values())[0].index[start_idx]
        end_date = list(stock_data.values())[0].index[-1]
        
        print(f"  Period: {start_date.date()} to {end_date.date()}")
        
        for date_idx in range(start_idx, len(list(stock_data.values())[0])):
            current_date = list(stock_data.values())[0].index[date_idx]
            
            # Check exits first
            for symbol in list(self.positions.keys()):
                if symbol in stock_data:
                    df = stock_data[symbol]
                    if current_date in df.index:
                        current_price = df.loc[current_date, 'Close']
                        pos = self.positions[symbol]
                        
                        exit_reason = None
                        if pos['type'] == 'LONG':
                            if current_price <= pos['sl']: exit_reason = 'SL'
                            elif current_price >= pos['tp']: exit_reason = 'TP'
                        else:
                            if current_price >= pos['sl']: exit_reason = 'SL'
                            elif current_price <= pos['tp']: exit_reason = 'TP'
                        
                        if exit_reason:
                            proceeds = pos['shares'] * current_price
                            pnl = proceeds - (pos['shares'] * pos['entry_price'])
                            
                            self.trades.append({
                                'symbol': symbol,
                                'action': 'SELL',
                                'shares': pos['shares'],
                                'price': current_price,
                                'value': proceeds,
                                'date': current_date,
                                'pnl': pnl,
                                'status': 'CLOSED',
                                'exit_reason': exit_reason,
                                'holding_days': (current_date - pos['entry_date']).days
                            })
                            
                            self.cash += proceeds
                            print(f"  [{exit_reason}] {symbol} @ ${current_price:.2f} | PnL: ${pnl:.2f}")
                            del self.positions[symbol]
            
            # Check for new signals (limit 1 trade per day)
            if len(self.positions) < 3:
                for symbol in symbols:
                    if symbol in self.positions:
                        continue
                    if symbol not in stock_data:
                        continue
                    
                    df = stock_data[symbol]
                    if current_date in df.index:
                        signal = self.check_signal(df[:current_date])
                        
                        if signal == 'BUY':
                            current_price = df.loc[current_date, 'Close']
                            shares = int(self.cash * 0.1 / current_price)
                            
                            if shares > 0:
                                self.positions[symbol] = {
                                    'type': 'LONG',
                                    'shares': shares,
                                    'entry_price': current_price,
                                    'entry_date': current_date,
                                    'sl': current_price * 0.98,
                                    'tp': current_price * 1.06
                                }
                                self.cash -= shares * current_price
                                
                                self.trades.append({
                                    'symbol': symbol,
                                    'action': 'BUY',
                                    'shares': shares,
                                    'price': current_price,
                                    'value': shares * current_price,
                                    'date': current_date,
                                    'pnl': 0,
                                    'status': 'OPEN'
                                })
                                print(f"  [BUY] {symbol}: {shares} @ ${current_price:.2f}")
                                break
            
            # Record equity
            portfolio_value = self.cash
            for symbol, pos in self.positions.items():
                if symbol in stock_data:
                    current_price = stock_data[symbol].loc[current_date, 'Close']
                    portfolio_value += pos['shares'] * current_price
            
            self.equity_curve.append({
                'date': current_date,
                'cash': self.cash,
                'positions_value': portfolio_value - self.cash,
                'total_value': portfolio_value
            })
        
        # Close remaining positions at end
        print("\n[Closing remaining positions...]")
        
        for symbol, pos in list(self.positions.items()):
            if symbol in stock_data:
                final_price = stock_data[symbol].iloc[-1]['Close']
                proceeds = pos['shares'] * final_price
                pnl = proceeds - (pos['shares'] * pos['entry_price'])
                
                self.trades.append({
                    'symbol': symbol,
                    'action': 'SELL',
                    'shares': pos['shares'],
                    'price': final_price,
                    'value': proceeds,
                    'date': stock_data[symbol].index[-1],
                    'pnl': pnl,
                    'status': 'CLOSED',
                    'exit_reason': 'END',
                    'holding_days': (stock_data[symbol].index[-1] - pos['entry_date']).days
                })
                
                self.cash += proceeds
                print(f"  [CLOSE] {symbol}: {pos['shares']} @ ${final_price:.2f} | PnL: ${pnl:.2f}")
        
        return self.generate_summary()
    
    def generate_summary(self) -> dict:
        """Generate trading summary."""
        closed_trades = [t for t in self.trades if t['status'] == 'CLOSED']
        
        if not closed_trades:
            return {'trades': self.trades}
        
        wins = [t for t in closed_trades if t['pnl'] > 0]
        losses = [t for t in closed_trades if t['pnl'] <= 0]
        
        win_rate = len(wins) / len(closed_trades) * 100 if closed_trades else 0
        
        total_pnl = sum([t['pnl'] for t in closed_trades])
        avg_win = sum([t['pnl'] for t in wins]) / len(wins) if wins else 0
        avg_loss = sum([t['pnl'] for t in losses]) / len(losses) if losses else 0
        
        # Calculate max drawdown
        equity = [e['total_value'] for e in self.equity_curve]
        peak = equity[0]
        max_dd = 0
        
        for val in equity:
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            if dd > max_dd:
                max_dd = dd
        
        # Calculate streaks
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        
        for t in closed_trades:
            if t['pnl'] > 0:
                current_streak = current_streak + 1 if current_streak > 0 else 1
                max_win_streak = max(max_win_streak, current_streak)
            else:
                current_streak = current_streak - 1 if current_streak < 0 else -1
                max_loss_streak = max(max_loss_streak, abs(current_streak))
        
        print("\n" + "=" * 70)
        print("  PAPER TRADING RESULTS")
        print("=" * 70)
        
        print(f"""
    +---------------------------------------------------------------+
    |  TRADING SUMMARY                                             |
    +---------------------------------------------------------------+
    |  Total Trades:        {len(closed_trades):>10}                             |
    |  Winning Trades:      {len(wins):>10}                             |
    |  Losing Trades:       {len(losses):>10}                             |
    |  Win Rate:           {win_rate:>10.1f}%                            |
    +---------------------------------------------------------------+
    |  PROFIT/LOSS                                                  |
    +---------------------------------------------------------------+
    |  Total PnL:         ${total_pnl:>10,.2f}                        |
    |  Average Win:       ${avg_win:>10,.2f}                        |
    |  Average Loss:     ${avg_loss:>10,.2f}                        |
    |  Final Capital:      ${self.cash:>10,.2f}                        |
    +---------------------------------------------------------------+
    |  RISK METRICS                                               |
    +---------------------------------------------------------------+
    |  Max Drawdown:      {max_dd*100:>10.2f}%                            |
    |  Best Trade:       ${max([t['pnl'] for t in closed_trades]):>10,.2f}                        |
    |  Worst Trade:      ${min([t['pnl'] for t in closed_trades]):>10,.2f}                        |
    +---------------------------------------------------------------+
    |  STREAKS                                                     |
    +---------------------------------------------------------------+
    |  Max Win Streak:    {max_win_streak:>10}                             |
    |  Max Loss Streak:    {max_loss_streak:>10}                             |
    +---------------------------------------------------------------+
        """)
        
        # Trade log
        print("\n[Trade Log]")
        print("-" * 70)
        
        for i, t in enumerate(closed_trades, 1):
            pnl_str = f"+${t['pnl']:.2f}" if t['pnl'] > 0 else f"-${abs(t['pnl']):.2f}"
            print(f"{i:2}. {t['symbol']:<6} {t['action']:<4} ${t['price']:>8.2f} | {pnl_str:>10} | {t.get('exit_reason', 'N/A')}")
        
        return {
            'total_trades': len(closed_trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'final_capital': self.cash,
            'max_drawdown': max_dd * 100,
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak,
            'trades': closed_trades,
            'equity_curve': self.equity_curve
        }


def main():
    print("\n" + "=" * 70)
    print("  STEP 5: PAPER TRADING SIMULATION")
    print("=" * 70)
    
    # Portfolio stocks
    symbols = ['GOOGL', 'META', 'AAPL', 'SPY', 'MSFT']
    
    # Run paper trading
    trader = PaperTrader(initial_capital=100000)
    results = trader.run_paper_trading(symbols, weeks=4)
    
    return results


if __name__ == "__main__":
    main()