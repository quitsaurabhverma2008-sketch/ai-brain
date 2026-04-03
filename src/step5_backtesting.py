"""
Step 5: Backtesting System
=================================
Test strategy on historical data
"""

import pandas as pd
import numpy as np


class BacktestingSystem:
    """
    Backtesting System - Test strategy on historical data
    """
    
    def __init__(self, capital: float = 10000, risk_per_trade: float = 0.02):
        self.initial_capital = capital
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.trades = []
        self.positions = []
        self.current_position = None
    
    # ========================
    # GENERATE SIGNALS
    # ========================
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals for entire dataset."""
        df = df.copy()
        df['Signal'] = 'HOLD'
        
        # Generate signals
        for i in range(50, len(df)):
            latest = df.iloc[:i+1].iloc[-1]
            prev = df.iloc[:i+1].iloc[-2]
            
            # Buy conditions
            buy_score = 0
            if latest['RSI'] < 35: buy_score += 1
            if latest['MACD'] > latest['MACD_Signal']: buy_score += 1
            if prev['MACD'] <= prev['MACD_Signal'] and latest['MACD'] > latest['MACD_Signal']: buy_score += 1
            if latest['Close'] > latest['SMA_20']: buy_score += 1
            if latest['SMA_20'] > latest['SMA_50']: buy_score += 1
            
            # Sell conditions
            sell_score = 0
            if latest['RSI'] > 65: sell_score += 1
            if latest['MACD'] < latest['MACD_Signal']: sell_score += 1
            if prev['MACD'] >= prev['MACD_Signal'] and latest['MACD'] < latest['MACD_Signal']: sell_score += 1
            if latest['Close'] < latest['SMA_20']: sell_score += 1
            if latest['SMA_20'] < latest['SMA_50']: sell_score += 1
            
            if buy_score >= 3:
                df.loc[df.index[i], 'Signal'] = 'BUY'
            elif sell_score >= 3:
                df.loc[df.index[i], 'Signal'] = 'SELL'
            else:
                df.loc[df.index[i], 'Signal'] = 'HOLD'
        
        return df
    
    # ========================
    # RUN BACKTEST
    # ========================
    
    def run_backtest(self, df: pd.DataFrame) -> dict:
        """Run backtest on historical data."""
        df = self.generate_signals(df)
        
        # Process each signal
        for i in range(len(df)):
            row = df.iloc[i]
            signal = row['Signal']
            
            if signal == 'BUY' and self.current_position is None:
                # Open BUY position
                entry_price = row['Close']
                sl_price = entry_price * 0.98  # 2% stop loss
                tp_price = entry_price * 1.06  # 6% take profit
                
                position_size = self.capital / entry_price
                self.current_position = {
                    'type': 'LONG',
                    'entry': entry_price,
                    'sl': sl_price,
                    'tp': tp_price,
                    'size': position_size,
                    'entry_date': str(row.name)[:10]
                }
                
            elif signal == 'SELL' and self.current_position is None:
                # Open SELL position
                entry_price = row['Close']
                sl_price = entry_price * 1.02
                tp_price = entry_price * 0.94
                
                position_size = self.capital / entry_price
                self.current_position = {
                    'type': 'SHORT',
                    'entry': entry_price,
                    'sl': sl_price,
                    'tp': tp_price,
                    'size': position_size,
                    'entry_date': str(row.name)[:10]
                }
                
            elif self.current_position is not None:
                # Check for exit
                current_price = row['Close']
                pos = self.current_position
                
                if pos['type'] == 'LONG':
                    # Check SL or TP hit
                    if current_price <= pos['sl']:
                        self._close_trade('STOP_LOSS', current_price, row.name)
                    elif current_price >= pos['tp']:
                        self._close_trade('TAKE_PROFIT', current_price, row.name)
                    elif signal == 'SELL':  # Exit signal
                        self._close_trade('SIGNAL', current_price, row.name)
                        
                else:  # SHORT
                    if current_price >= pos['sl']:
                        self._close_trade('STOP_LOSS', current_price, row.name)
                    elif current_price <= pos['tp']:
                        self._close_trade('TAKE_PROFIT', current_price, row.name)
                    elif signal == 'BUY':  # Exit signal
                        self._close_trade('SIGNAL', current_price, row.name)
        
        # Close any open position at end
        if self.current_position is not None:
            self._close_trade('END', df.iloc[-1]['Close'], df.iloc[-1].name)
        
        return self.calculate_metrics()
    
    def _close_trade(self, exit_reason: str, exit_price: float, exit_date):
        """Close a trade and record it."""
        pos = self.current_position
        
        if pos['type'] == 'LONG':
            pnl = (exit_price - pos['entry']) * pos['size']
        else:
            pnl = (pos['entry'] - exit_price) * pos['size']
        
        pnl_percent = (pnl / self.capital) * 100
        
        self.trades.append({
            'type': pos['type'],
            'entry_price': pos['entry'],
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'exit_reason': exit_reason,
            'entry_date': pos['entry_date'],
            'exit_date': str(exit_date)[:10]
        })
        
        self.capital += pnl
        self.current_position = None
    
    # ========================
    # CALCULATE METRICS
    # ========================
    
    def calculate_metrics(self) -> dict:
        """Calculate backtest metrics."""
        if not self.trades:
            return {'error': 'No trades generated'}
        
        pnls = [t['pnl'] for t in self.trades]
        wins = [t for t in self.trades if t['pnl'] > 0]
        losses = [t for t in self.trades if t['pnl'] <= 0]
        
        return {
            'total_trades': len(self.trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': (len(wins) / len(self.trades)) * 100,
            'total_pnl': sum(pnls),
            'total_return': ((self.capital - self.initial_capital) / self.initial_capital) * 100,
            'avg_win': sum([t['pnl'] for t in wins]) / len(wins) if wins else 0,
            'avg_loss': sum([t['pnl'] for t in losses]) / len(losses) if losses else 0,
            'best_trade': max(pnls),
            'worst_trade': min(pnls),
            'final_capital': self.capital
        }
    
    def print_results(self, metrics: dict):
        """Print backtest results."""
        print("\n" + "=" * 60)
        print("  BACKTESTING RESULTS")
        print("=" * 60)
        
        print(f"""
    +---------------------------------------------------------------+
    |  OVERALL PERFORMANCE                                          |
    +---------------------------------------------------------------+
    |  Total Trades:       {metrics['total_trades']:>10}                             |
    |  Winning Trades:    {metrics['winning_trades']:>10}                             |
    |  Losing Trades:     {metrics['losing_trades']:>10}                             |
    |  Win Rate:          {metrics['win_rate']:>10.1f}%                            |
    +---------------------------------------------------------------+
    |  PROFIT/LOSS                                                   |
    +---------------------------------------------------------------+
    |  Total PnL:        ${metrics['total_pnl']:>10,.2f}                          |
    |  Total Return:     {metrics['total_return']:>10.2f}%                           |
    |  Final Capital:     ${metrics['final_capital']:>10,.2f}                          |
    +---------------------------------------------------------------+
    |  TRADE STATISTICS                                            |
    +---------------------------------------------------------------+
    |  Average Win:      ${metrics['avg_win']:>10,.2f}                          |
    |  Average Loss:     ${metrics['avg_loss']:>10,.2f}                          |
    |  Best Trade:       ${metrics['best_trade']:>10,.2f}                          |
    |  Worst Trade:      ${metrics['worst_trade']:>10,.2f}                          |
    +---------------------------------------------------------------+
        """)
        
        # Print trade log
        print("\n[Trade Log]:")
        print("-" * 60)
        for i, trade in enumerate(self.trades[:10], 1):
            pnl_str = f"${trade['pnl']:,.2f}" if trade['pnl'] > 0 else f"-${abs(trade['pnl']):,.2f}"
            print(f"{i}. {trade['type']:<5} | Entry: ${trade['entry_price']:>8,.0f} | "
                  f"Exit: ${trade['exit_price']:>8,.0f} | "
                  f"PnL: {pnl_str:<10} | {trade['exit_reason']}")
        
        if len(self.trades) > 10:
            print(f"... and {len(self.trades) - 10} more trades")
        
        # Risk metrics
        print("\n" + "=" * 60)
        print("  RISK METRICS")
        print("=" * 60)
        
        # Calculate max drawdown
        peak = self.initial_capital
        max_dd = 0
        capital_curve = [self.initial_capital]
        
        for trade in self.trades:
            capital_curve.append(capital_curve[-1] + trade['pnl'])
        
        for cap in capital_curve:
            if cap > peak:
                peak = cap
            dd = (peak - cap) / peak
            if dd > max_dd:
                max_dd = dd
        
        print(f"""
    |  Max Drawdown:      {max_dd*100:.2f}%                                     |
    |  Risk:Reward:      1:3.0 (fixed)                                |
    |  Risk per Trade:    2.00%                                      |
    +---------------------------------------------------------------+
        """)
    
    def print_trade_summary(self):
        """Print last 5 trades summary."""
        print("\n[Last 5 Trades]:")
        print("-" * 60)
        for trade in self.trades[-5:]:
            pnl_str = f"+${trade['pnl']:.2f}" if trade['pnl'] > 0 else f"-${abs(trade['pnl']):.2f}"
            print(f"{trade['type']:<6} {trade['entry_date']} -> {trade['exit_date']} | "
                  f"${trade['entry_price']:>8,.0f} -> ${trade['exit_price']:>8,.0f} | "
                  f"{pnl_str:>10} ({trade['exit_reason']})")


def main():
    print("\n" + "=" * 60)
    print("  STEP 5: BACKTESTING SYSTEM")
    print("=" * 60)
    
    # Load data
    print("\n[1] Loading data with indicators...")
    df = pd.read_csv('D:/saurabh/ai brain/data/BTC_USD_with_indicators.csv', 
                    parse_dates=True, index_col=0)
    print(f"    Loaded {len(df)} rows")
    
    # Run backtest
    print("\n[2] Running backtest...")
    backtest = BacktestingSystem(capital=10000, risk_per_trade=0.02)
    metrics = backtest.run_backtest(df)
    
    # Print results
    backtest.print_results(metrics)
    backtest.print_trade_summary()
    
    return metrics


if __name__ == "__main__":
    main()