"""
Step 5: Risk Analysis & Stress Testing
=====================================
"""

import pandas as pd
import numpy as np
import yfinance as yf


class RiskAnalyzer:
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
    
    def download_data(self, symbol: str, start: str = None, end: str = None):
        stock = yf.Ticker(symbol)
        if start and end:
            df = stock.history(start=start, end=end, interval="1d")
        else:
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
    
    def run_backtest(self, df: pd.DataFrame, allocated_capital: float) -> dict:
        capital = allocated_capital
        trades = []
        equity_curve = [capital]
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
                    trades.append({
                        'pnl': pnl,
                        'pnl_pct': (pnl / capital) * 100,
                        'type': p['type'],
                        'exit': exit_reason,
                        'date': str(df.index[i])[:10]
                    })
                    capital += pnl
                    equity_curve.append(capital)
                    position = None
        
        if position:
            price = df.iloc[-1]['Close']
            p = position
            if p['type'] == 'LONG':
                pnl = (price - p['entry']) * (capital / p['entry'])
            else:
                pnl = (p['entry'] - price) * (capital / p['entry'])
            trades.append({'pnl': pnl, 'pnl_pct': (pnl/capital)*100, 'type': p['type'], 'exit': 'END'})
            capital += pnl
            equity_curve.append(capital)
        
        return {
            'trades': trades,
            'equity_curve': equity_curve,
            'final_capital': capital,
            'return_pct': ((capital - allocated_capital) / allocated_capital) * 100
        }
    
    def calculate_risk_metrics(self, trades: list, equity_curve: list) -> dict:
        if not trades:
            return {}
        
        pnls = [t['pnl'] for t in trades]
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] < 0]
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Returns
        total_return = sum(pnls)
        avg_win = sum([t['pnl'] for t in wins]) / len(wins) if wins else 0
        avg_loss = sum([t['pnl'] for t in losses]) / len(losses) if losses else 0
        
        # Profit Factor
        gross_profit = sum([t['pnl'] for t in wins]) if wins else 0
        gross_loss = abs(sum([t['pnl'] for t in losses])) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Max Drawdown
        peak = self.initial_capital
        max_dd = 0
        max_dd_pct = 0
        for val in equity_curve:
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd * 100
        
        # Daily returns for Sharpe
        daily_returns = []
        for i in range(1, len(equity_curve)):
            daily_returns.append((equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1])
        
        avg_daily_return = np.mean(daily_returns) if daily_returns else 0
        std_daily_return = np.std(daily_returns) if daily_returns else 0
        
        # Sharpe Ratio (annualized, assuming 252 trading days)
        if std_daily_return > 0:
            sharpe_ratio = (avg_daily_return / std_daily_return) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Expectancy
        expectancy = (win_rate/100 * avg_win) + ((1-win_rate/100) * avg_loss) if total_trades > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_dd_pct,
            'sharpe_ratio': sharpe_ratio,
            'expectancy': expectancy,
            'avg_daily_return': avg_daily_return * 100,
            'std_daily_return': std_daily_return * 100
        }
    
    def run_period_test(self, symbol: str, start: str, end: str, allocation: float) -> dict:
        """Run backtest for specific time period."""
        df = self.download_data(symbol, start, end)
        if df is None or len(df) < 50:
            return {'error': 'No data'}
        
        df = self.calculate_indicators(df)
        result = self.run_backtest(df, allocation)
        metrics = self.calculate_risk_metrics(result['trades'], result['equity_curve'])
        
        return {
            'period': f"{start} to {end}",
            'data_points': len(df),
            'result': result,
            'metrics': metrics
        }


def main():
    print("\n" + "=" * 70)
    print("  RISK ANALYSIS - AGGRESSIVE PORTFOLIO")
    print("=" * 70)
    
    analyzer = RiskAnalyzer(initial_capital=100000)
    
    # Aggressive portfolio allocation
    portfolio = {
        'GOOGL': 40000,
        'META': 35000,
        'AAPL': 10000,
        'SPY': 10000,
        'MSFT': 5000,
    }
    
    print("\n[Running backtests on portfolio stocks...]")
    
    all_trades = []
    all_equity = []
    
    for symbol, allocation in portfolio.items():
        print(f"[{symbol}] Processing...", end=" ")
        df = analyzer.download_data(symbol)
        if df is None or len(df) < 50:
            print("FAILED")
            continue
        
        df = analyzer.calculate_indicators(df)
        result = analyzer.run_backtest(df, allocation)
        
        # Add to portfolio
        for trade in result['trades']:
            trade['symbol'] = symbol
            all_trades.append(trade)
        
        print(f"Return: {result['return_pct']:.2f}%")
    
    # Calculate combined equity curve
    all_equity = []
    for symbol, allocation in portfolio.items():
        df = analyzer.download_data(symbol)
        if df is None:
            continue
        df = analyzer.calculate_indicators(df)
        result = analyzer.run_backtest(df, allocation)
        all_equity.append(result['equity_curve'])
    
    # Find minimum length
    min_len = min([len(e) for e in all_equity])
    combined_equity = []
    for i in range(min_len):
        total = sum([e[i] for e in all_equity])
        combined_equity.append(total)
    
    # Calculate risk metrics
    metrics = analyzer.calculate_risk_metrics(all_trades, combined_equity)
    
    print("\n" + "=" * 70)
    print("  RISK METRICS")
    print("=" * 70)
    
    print(f"""
    +---------------------------------------------------------------+
    |  BASIC METRICS                                                |
    +---------------------------------------------------------------+
    |  Total Trades:        {metrics['total_trades']:>10}                             |
    |  Winning Trades:      {metrics['winning_trades']:>10}                             |
    |  Losing Trades:       {metrics['losing_trades']:>10}                             |
    |  Win Rate:           {metrics['win_rate']:>10.1f}%                            |
    +---------------------------------------------------------------+
    |  PROFIT METRICS                                              |
    +---------------------------------------------------------------+
    |  Total Return:       ${metrics['total_return']:>10,.2f}                       |
    |  Average Win:        ${metrics['avg_win']:>10,.2f}                       |
    |  Average Loss:      ${metrics['avg_loss']:>10,.2f}                       |
    |  Profit Factor:      {metrics['profit_factor']:>10.2f}                             |
    |  Expectancy:        ${metrics['expectancy']:>10,.2f}                       |
    +---------------------------------------------------------------+
    |  RISK METRICS                                                |
    +---------------------------------------------------------------+
    |  Max Drawdown:      {metrics['max_drawdown']:>10.2f}%                            |
    |  Sharpe Ratio:       {metrics['sharpe_ratio']:>10.2f}                             |
    |  Avg Daily Return:  {metrics['avg_daily_return']:>10.3f}%                           |
    |  Std Daily Return:  {metrics['std_daily_return']:>10.3f}%                           |
    +---------------------------------------------------------------+
    """)
    
    # Trade analysis
    print("\n" + "=" * 70)
    print("  TRADE ANALYSIS")
    print("=" * 70)
    
    print(f"\n[Win/Loss Breakdown]")
    print(f"  Wins: {metrics['winning_trades']} ({metrics['win_rate']:.1f}%)")
    print(f"  Losses: {metrics['losing_trades']} ({100-metrics['win_rate']:.1f}%)")
    
    print(f"\n[Win/Loss Ratio]")
    if metrics['avg_loss'] != 0:
        win_loss_ratio = abs(metrics['avg_win'] / metrics['avg_loss'])
        print(f"  Win/Loss Ratio: {win_loss_ratio:.2f}")
    
    print(f"\n[Profit Factor]")
    print(f"  Gross Profit: ${sum([t['pnl'] for t in all_trades if t['pnl']>0]):,.2f}")
    print(f"  Gross Loss: ${abs(sum([t['pnl'] for t in all_trades if t['pnl']<0])):,.2f}")
    print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    
    print("\n" + "=" * 70)
    print("  PERIOD ANALYSIS")
    print("=" * 70)
    
    # Test different periods
    periods = [
        ("2020-01-01", "2021-12-31", "2020-2021 (Bull)"),
        ("2022-01-01", "2022-12-31", "2022 (Bear)"),
        ("2023-01-01", "2024-12-31", "2023-2024 (Recovery)"),
    ]
    
    period_results = []
    
    for start, end, name in periods:
        print(f"\n[{name}]")
        
        period_trades = []
        
        for symbol, allocation in portfolio.items():
            result = analyzer.run_period_test(symbol, start, end, allocation)
            if 'error' not in result:
                for trade in result['result']['trades']:
                    trade['symbol'] = symbol
                    period_trades.append(trade)
        
        if period_trades:
            period_metrics = analyzer.calculate_risk_metrics(period_trades, [100000])
            print(f"  Trades: {period_metrics['total_trades']}")
            print(f"  Return: {period_metrics['total_return']:.2f}%")
            print(f"  Win Rate: {period_metrics['win_rate']:.1f}%")
            print(f"  Max DD: {period_metrics['max_drawdown']:.2f}%")
            
            period_results.append({
                'period': name,
                'metrics': period_metrics
            })
    
    # Summary
    print("\n" + "=" * 70)
    print("  RISK ASSESSMENT SUMMARY")
    print("=" * 70)
    
    print(f"""
    [IS THE PORTFOLIO STABLE?]
    
    Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
    - 0-1: Poor
    - 1-2: Acceptable  
    - 2-3: Good
    - 3+: Excellent
    
    Your Sharpe: {metrics['sharpe_ratio']:.2f}
    
    Max Drawdown: {metrics['max_drawdown']:.2f}%
    - <10%: Excellent
    - 10-20%: Good
    - 20-30%: Acceptable
    - 30%+: High Risk
    
    Your Max DD: {metrics['max_drawdown']:.2f}%
    
    Profit Factor: {metrics['profit_factor']:.2f}
    - <1: Losing money
    - 1-1.5: Marginal
    - 1.5-2: Good
    - 2+: Excellent
    
    Your PF: {metrics['profit_factor']:.2f}
    """)
    
    return metrics


if __name__ == "__main__":
    main()