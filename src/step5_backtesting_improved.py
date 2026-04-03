"""
Step 5: IMPROVED Backtesting System
=================================
With Trend Filter, Strict Signals, Cooldown, Better Risk Management
"""

import pandas as pd
import numpy as np


class ImprovedBacktestingSystem:
    """
    Improved Backtesting System with better filters
    """
    
    def __init__(self, capital: float = 10000, risk_per_trade: float = 0.02):
        self.initial_capital = capital
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.trades = []
        self.current_position = None
        self.last_trade_date = None
        self.cooldown_hours = 6  # Minimum 6 hours between trades
    
    # ========================
    # GENERATE IMPROVED SIGNALS
    # ========================
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate filtered signals."""
        df = df.copy()
        df['Signal'] = 'HOLD'
        df['Trend'] = 'NEUTRAL'
        
        # Determine trend using SMA-200 equivalent (50 for hourly data)
        df['Trend'] = df.apply(
            lambda x: 'UPTREND' if x['SMA_20'] > x['SMA_50'] else 'DOWNTREND',
            axis=1
        )
        
        # Generate signals
        for i in range(50, len(df)):
            latest = df.iloc[:i+1].iloc[-1]
            prev = df.iloc[:i+1].iloc[-2]
            trend = latest['Trend']
            
            # Calculate scores
            buy_score = 0
            sell_score = 0
            
            # RSI (strong signal only)
            if latest['RSI'] < 30: buy_score += 2  # Strong oversold
            elif latest['RSI'] < 35: buy_score += 1
            if latest['RSI'] > 70: sell_score += 2  # Strong overbought
            elif latest['RSI'] > 65: sell_score += 1
            
            # MACD Crossover (recent)
            if (prev['MACD'] <= prev['MACD_Signal'] and 
                latest['MACD'] > latest['MACD_Signal']):
                buy_score += 2
            elif (prev['MACD'] >= prev['MACD_Signal'] and 
                  latest['MACD'] < latest['MACD_Signal']):
                sell_score += 2
            
            # MACD momentum (always adds)
            if latest['MACD'] > latest['MACD_Signal']: buy_score += 1
            if latest['MACD'] < latest['MACD_Signal']: sell_score += 1
            
            # Price vs Moving Averages
            if latest['Close'] > latest['SMA_20']: buy_score += 1
            if latest['Close'] < latest['SMA_20']: sell_score += 1
            
            # Strong trend confirmation
            if trend == 'UPTREND': buy_score += 1
            if trend == 'DOWNTREND': sell_score += 1
            
            # Bollinger extreme (only for entry)
            if latest['Close'] < latest['BB_Lower']: buy_score += 1
            if latest['Close'] > latest['BB_Upper']: sell_score += 1
            
            # STRICT: Require 4+ conditions
            # PLUS: Trend must match direction
            if buy_score >= 4 and trend == 'UPTREND':
                df.loc[df.index[i], 'Signal'] = 'BUY'
            elif sell_score >= 4 and trend == 'DOWNTREND':
                df.loc[df.index[i], 'Signal'] = 'SELL'
            else:
                df.loc[df.index[i], 'Signal'] = 'HOLD'
        
        return df
    
    # ========================
    # RUN BACKTEST WITH COOLDOWN
    # ========================
    
    def run_backtest(self, df: pd.DataFrame) -> dict:
        """Run backtest with cooldown."""
        df = self.generate_signals(df)
        
        for i in range(len(df)):
            row = df.iloc[i]
            signal = row['Signal']
            current_date = row.name
            
            # Check cooldown
            if self.last_trade_date is not None:
                hours_diff = (current_date - self.last_trade_date).total_seconds() / 3600
                if hours_diff < self.cooldown_hours:
                    continue
            
            if signal == 'BUY' and self.current_position is None:
                self._open_position('LONG', row['Close'], 
                                  row['Close'] * 0.98,  # 2% SL
                                  row['Close'] * 1.06,  # 6% TP
                                  current_date)
                
            elif signal == 'SELL' and self.current_position is None:
                self._open_position('SHORT', row['Close'],
                                  row['Close'] * 1.02,
                                  row['Close'] * 0.94,
                                  current_date)
                
            elif self.current_position is not None:
                self._check_exit(row['Close'], current_date, signal)
        
        # Close open position
        if self.current_position is not None:
            self._close_trade('END', df.iloc[-1]['Close'], df.iloc[-1].name)
        
        return self.calculate_metrics()
    
    def _open_position(self, ptype, entry, sl, tp, date):
        """Open a position."""
        position_size = self.capital / entry
        self.current_position = {
            'type': ptype,
            'entry': entry,
            'sl': sl,
            'tp': tp,
            'size': position_size,
            'entry_date': date
        }
    
    def _check_exit(self, price, date, signal):
        """Check if should exit position."""
        pos = self.current_position
        
        if pos['type'] == 'LONG':
            if price <= pos['sl']:
                self._close_trade('STOP_LOSS', price, date)
            elif price >= pos['tp']:
                self._close_trade('TAKE_PROFIT', price, date)
            elif signal == 'SELL':  # Exit on opposite signal
                self._close_trade('SIGNAL', price, date)
        else:
            if price >= pos['sl']:
                self._close_trade('STOP_LOSS', price, date)
            elif price <= pos['tp']:
                self._close_trade('TAKE_PROFIT', price, date)
            elif signal == 'BUY':
                self._close_trade('SIGNAL', price, date)
    
    def _close_trade(self, reason, price, date):
        """Close trade and record."""
        pos = self.current_position
        
        if pos['type'] == 'LONG':
            pnl = (price - pos['entry']) * pos['size']
        else:
            pnl = (pos['entry'] - price) * pos['size']
        
        self.trades.append({
            'type': pos['type'],
            'entry_price': pos['entry'],
            'exit_price': price,
            'pnl': pnl,
            'pnl_percent': (pnl / self.capital) * 100,
            'exit_reason': reason,
            'entry_date': str(pos['entry_date'])[:19],
            'exit_date': str(date)[:19]
        })
        
        self.capital += pnl
        self.last_trade_date = date
        self.current_position = None
    
    def calculate_metrics(self) -> dict:
        """Calculate metrics."""
        if not self.trades:
            return {'error': 'No trades'}
        
        wins = [t for t in self.trades if t['pnl'] > 0]
        losses = [t for t in self.trades if t['pnl'] <= 0]
        
        return {
            'total_trades': len(self.trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': (len(wins) / len(self.trades)) * 100,
            'total_pnl': sum([t['pnl'] for t in self.trades]),
            'total_return': ((self.capital - self.initial_capital) / self.initial_capital) * 100,
            'avg_win': sum([t['pnl'] for t in wins]) / len(wins) if wins else 0,
            'avg_loss': sum([t['pnl'] for t in losses]) / len(losses) if losses else 0,
            'best_trade': max([t['pnl'] for t in self.trades]),
            'worst_trade': min([t['pnl'] for t in self.trades]),
            'final_capital': self.capital
        }
    
    def print_results(self, metrics: dict):
        """Print results."""
        print("\n" + "=" * 70)
        print("  IMPROVED BACKTESTING RESULTS")
        print("=" * 70)
        
        print(f"""
    +---------------------------------------------------------------+
    |  IMPROVEMENTS APPLIED:                                         |
    |  - Trend Filter: Only trade with trend                            |
    |  - Strict Signals: 4+ conditions required                  |
    |  - Cooldown: 6 hours between trades                         |
    |  - Better Risk:Reward maintained                           |
    +---------------------------------------------------------------+
    |  OVERALL PERFORMANCE                                         |
    +---------------------------------------------------------------+
    |  Total Trades:       {metrics['total_trades']:>10}                             |
    |  Winning Trades:    {metrics['winning_trades']:>10}                             |
    |  Losing Trades:     {metrics['losing_trades']:>10}                             |
    |  Win Rate:          {metrics['win_rate']:>10.1f}%                            |
    +---------------------------------------------------------------+
    |  PROFIT/LOSS                                                |
    +---------------------------------------------------------------+
    |  Total PnL:         ${metrics['total_pnl']:>10,.2f}                          |
    |  Total Return:      {metrics['total_return']:>10.2f}%                          |
    |  Final Capital:     ${metrics['final_capital']:>10,.2f}                          |
    +---------------------------------------------------------------+
    |  TRADE STATISTICS                                           |
    +---------------------------------------------------------------+
    |  Average Win:     ${metrics['avg_win']:>10,.2f}                          |
    |  Average Loss:    ${metrics['avg_loss']:>10,.2f}                          |
    |  Best Trade:      ${metrics['best_trade']:>10,.2f}                          |
    |  Worst Trade:     ${metrics['worst_trade']:>10,.2f}                          |
    +---------------------------------------------------------------+
        """)
        
        print("\n[Trade Log]:")
        print("-" * 70)
        for i, t in enumerate(self.trades, 1):
            pnl_str = f"+${t['pnl']:.2f}" if t['pnl'] > 0 else f"-${abs(t['pnl']):.2f}"
            print(f"{i:2}. {t['type']:<5} | Entry: ${t['entry_price']:>8,.0f} | "
                  f"Exit: ${t['exit_price']:>8,.0f} | PnL: {pnl_str:>10} | {t['exit_reason']}")
        
        # Drawdown
        print("\n" + "=" * 70)
        print("  RISK METRICS")
        print("=" * 70)
        
        capital_curve = [self.initial_capital]
        for t in self.trades:
            capital_curve.append(capital_curve[-1] + t['pnl'])
        
        peak = self.initial_capital
        max_dd = 0
        for cap in capital_curve:
            if cap > peak:
                peak = cap
            dd = (peak - cap) / peak
            if dd > max_dd:
                max_dd = dd
        
        print(f"""
    |  Max Drawdown:      {max_dd*100:.2f}%                                     |
    |  Risk:Reward:    1:3.0 (fixed)                                |
    |  Risk per Trade:  2.00%                                      |
    +---------------------------------------------------------------+
        """)


def main():
    print("\n" + "=" * 70)
    print("  IMPROVED STRATEGY - BACKTESTING")
    print("=" * 70)
    
    # Load data
    print("\n[1] Loading data with indicators...")
    df = pd.read_csv('D:/saurabh/ai brain/data/BTC_USD_with_indicators.csv', 
                    parse_dates=True, index_col=0)
    print(f"    Loaded {len(df)} rows (1h timeframe)")
    
    # Run improved backtest
    print("\n[2] Running improved backtest...")
    print("    Improvements:")
    print("    - Trend Filter: ON (trade only with trend)")
    print("    - Strict Signals: 4+ conditions")
    print("    - Cooldown: 6 hours")
    print("    - 2% SL, 6% TP (1:3 RR)")
    
    backtest = ImprovedBacktestingSystem(capital=10000, risk_per_trade=0.02)
    metrics = backtest.run_backtest(df)
    
    # Print results
    backtest.print_results(metrics)
    
    print("\n" + "=" * 70)
    print("  COMPARISON: ORIGINAL vs IMPROVED")
    print("=" * 70)
    print("""
    | Metric          | Original  | Improved  | Change    |
    |----------------|----------|----------|----------|
    | Win Rate       |  26.9%  |  ***%   |   ***    |
    | Total Return  |  -9.08% |  ***%   |   ***    |
    | Total Trades  |     26   |    ***   |   ***    |
    | Max Drawdown |  13.40%  |  ***%   |   ***    |
    """)
    
    return metrics


if __name__ == "__main__":
    main()