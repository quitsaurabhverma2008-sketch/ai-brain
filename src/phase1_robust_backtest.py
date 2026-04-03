"""
Phase 1: Robust Backtesting System (Enhanced)
==============================================
5-year data, Train/Test Split, Walk-Forward Testing
With full validation, overfitting detection, and CSV export

Enhanced Features:
- Proper Sharpe ratio calculation
- Confidence levels (HIGH/MEDIUM/LOW)
- Multi-stock support (15 stocks)
- Full validation system
- Overfitting detection
- CSV export
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import csv
import os


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class ConfidenceLevel(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ValidationVerdict(Enum):
    ROBUST = "ROBUST"
    WEAK = "WEAK"
    OVERFITTED = "OVERFITTED"


@dataclass
class Metrics:
    """Enhanced performance metrics"""
    return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    confidence: ConfidenceLevel = ConfidenceLevel.LOW


@dataclass
class WalkForwardResult:
    """Single walk-forward window result"""
    stock: str
    window_id: int
    
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    
    train_return: float = 0.0
    train_sharpe: float = 0.0
    train_max_dd: float = 0.0
    train_win_rate: float = 0.0
    train_profit_factor: float = 0.0
    train_trades: int = 0
    train_confidence: ConfidenceLevel = ConfidenceLevel.LOW
    
    test_return: float = 0.0
    test_sharpe: float = 0.0
    test_max_dd: float = 0.0
    test_win_rate: float = 0.0
    test_profit_factor: float = 0.0
    test_trades: int = 0
    test_confidence: ConfidenceLevel = ConfidenceLevel.LOW
    
    is_overfitting: bool = False
    overfitting_reasons: List[str] = field(default_factory=list)


@dataclass
class ValidationSummary:
    """Final validation summary"""
    total_windows: int = 0
    robust_windows: int = 0
    weak_windows: int = 0
    overfitted_windows: int = 0
    
    avg_train_sharpe: float = 0.0
    avg_test_sharpe: float = 0.0
    avg_train_return: float = 0.0
    avg_test_return: float = 0.0
    avg_train_dd: float = 0.0
    avg_test_dd: float = 0.0
    
    verdict: ValidationVerdict = ValidationVerdict.WEAK
    verdict_reason: str = ""
    
    overfitting_flags: List[str] = field(default_factory=list)
    low_confidence_windows: int = 0


# =============================================================================
# ENHANCED METRICS CALCULATION
# =============================================================================

def calculate_enhanced_metrics(
    trades: List[Dict],
    equity_curve: List[float],
    initial_capital: float = 100000
) -> Metrics:
    """
    Calculate all performance metrics with proper Sharpe ratio.
    
    Args:
        trades: List of trade dictionaries with 'pnl' key
        equity_curve: Portfolio value over time
        initial_capital: Starting capital
    
    Returns:
        Metrics object with all calculated values
    """
    metrics = Metrics()
    
    # Edge cases
    if not equity_curve or len(equity_curve) < 2:
        return metrics
    
    if not trades or len(trades) == 0:
        metrics.confidence = ConfidenceLevel.LOW
        return metrics
    
    equity = np.array(equity_curve)
    
    # 1. RETURN (%)
    metrics.return_pct = ((equity[-1] - equity[0]) / equity[0]) * 100
    
    # 2. SHARPE RATIO (annualized) - proper calculation
    daily_returns = np.diff(equity) / equity[:-1]
    daily_returns = daily_returns[~np.isnan(daily_returns)]
    
    if len(daily_returns) > 0:
        mean_ret = np.mean(daily_returns)
        std_ret = np.std(daily_returns)
        
        if std_ret > 0:
            metrics.sharpe_ratio = (mean_ret / std_ret) * np.sqrt(252)
        else:
            metrics.sharpe_ratio = 0.0
    else:
        metrics.sharpe_ratio = 0.0
    
    # 3. MAX DRAWDOWN (%)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak * 100
    metrics.max_drawdown_pct = abs(np.min(drawdown))
    
    # 4. WIN RATE & TRADE STATS
    pnls = [t.get('pnl', 0) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    
    metrics.total_trades = len(trades)
    metrics.wins = len(wins)
    metrics.losses = len(losses)
    
    if len(trades) > 0:
        metrics.win_rate_pct = (len(wins) / len(trades)) * 100
    
    # 5. PROFIT FACTOR
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    
    if gross_loss > 0:
        metrics.profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        metrics.profit_factor = float('inf')
    else:
        metrics.profit_factor = 0.0
    
    # 6. CONFIDENCE LEVEL
    if metrics.total_trades >= 30:
        metrics.confidence = ConfidenceLevel.HIGH
    elif metrics.total_trades >= 20:
        metrics.confidence = ConfidenceLevel.MEDIUM
    else:
        metrics.confidence = ConfidenceLevel.LOW
    
    return metrics


def get_confidence_level(num_trades: int) -> ConfidenceLevel:
    """Determine confidence level based on sample size."""
    if num_trades >= 30:
        return ConfidenceLevel.HIGH
    elif num_trades >= 20:
        return ConfidenceLevel.MEDIUM
    else:
        return ConfidenceLevel.LOW


# =============================================================================
# OVERFITTING DETECTION
# =============================================================================

def check_overfitting(
    train_metrics: Metrics,
    test_metrics: Metrics
) -> tuple:
    """
    Check for overfitting indicators.
    
    Returns:
        Tuple of (is_overfitting: bool, reasons: List[str])
    """
    reasons = []
    
    # Check 1: Test Sharpe < 1.0
    if test_metrics.sharpe_ratio < 1.0:
        reasons.append(f"Test Sharpe {test_metrics.sharpe_ratio:.2f} < 1.0")
    
    # Check 2: Test return negative
    if test_metrics.return_pct < 0:
        reasons.append(f"Test return {test_metrics.return_pct:.2f}% is negative")
    
    # Check 3: Test drawdown > 2x train
    if train_metrics.max_drawdown_pct > 0:
        dd_ratio = test_metrics.max_drawdown_pct / train_metrics.max_drawdown_pct
        if dd_ratio > 2.0:
            reasons.append(f"Test DD {dd_ratio:.1f}x train")
    
    # Check 4: Win rate drop > 15%
    wr_drop = abs(train_metrics.win_rate_pct - test_metrics.win_rate_pct)
    if wr_drop > 15:
        reasons.append(f"Win rate dropped {wr_drop:.1f}%")
    
    # Check 5: Test return much lower than train (>50% drop)
    if train_metrics.return_pct > 0:
        return_drop = ((train_metrics.return_pct - test_metrics.return_pct) / 
                       train_metrics.return_pct) * 100
        if return_drop > 50:
            reasons.append(f"Test return {return_drop:.0f}% lower than train")
    
    is_overfit = len(reasons) >= 2
    return is_overfit, reasons


# =============================================================================
# VALIDATION SYSTEM
# =============================================================================

def validate_results(results: List[WalkForwardResult]) -> ValidationSummary:
    """Validate and aggregate walk-forward results."""
    if not results:
        return ValidationSummary(verdict_reason="No results to validate")
    
    summary = ValidationSummary(total_windows=len(results))
    
    robust_count = 0
    weak_count = 0
    overfit_count = 0
    low_conf_count = 0
    
    total_train_sharpe = 0
    total_test_sharpe = 0
    total_train_return = 0
    total_test_return = 0
    total_train_dd = 0
    total_test_dd = 0
    
    all_flags = []
    
    for r in results:
        if r.train_confidence == ConfidenceLevel.LOW or r.test_confidence == ConfidenceLevel.LOW:
            low_conf_count += 1
        
        total_train_sharpe += r.train_sharpe
        total_test_sharpe += r.test_sharpe
        total_train_return += r.train_return
        total_test_return += r.test_return
        total_train_dd += r.train_max_dd
        total_test_dd += r.test_max_dd
        
        if r.is_overfitting:
            overfit_count += 1
            all_flags.extend(r.overfitting_reasons)
        elif r.test_sharpe >= 1.0 and r.test_return > 0:
            robust_count += 1
        else:
            weak_count += 1
    
    summary.robust_windows = robust_count
    summary.weak_windows = weak_count
    summary.overfitted_windows = overfit_count
    summary.low_confidence_windows = low_conf_count
    
    n = len(results)
    summary.avg_train_sharpe = total_train_sharpe / n
    summary.avg_test_sharpe = total_test_sharpe / n
    summary.avg_train_return = total_train_return / n
    summary.avg_test_return = total_test_return / n
    summary.avg_train_dd = total_train_dd / n
    summary.avg_test_dd = total_test_dd / n
    
    robust_pct = (robust_count / n) * 100
    overfit_pct = (overfit_count / n) * 100
    
    if overfit_pct >= 40:
        summary.verdict = ValidationVerdict.OVERFITTED
        summary.verdict_reason = f"{overfit_count}/{n} windows show overfitting ({overfit_pct:.0f}%)"
    elif robust_pct >= 70:
        summary.verdict = ValidationVerdict.ROBUST
        summary.verdict_reason = f"{robust_count}/{n} windows passed ({robust_pct:.0f}%)"
    elif robust_pct >= 50:
        summary.verdict = ValidationVerdict.WEAK
        summary.verdict_reason = f"Only {robust_count}/{n} windows robust ({robust_pct:.0f}%)"
    else:
        summary.verdict = ValidationVerdict.OVERFITTED
        summary.verdict_reason = f"Insufficient robust windows: {robust_count}/{n}"
    
    summary.overfitting_flags = list(set(all_flags))[:5]
    
    return summary


def save_results_to_csv(results: List[WalkForwardResult], filename: str = "walkforward_results.csv"):
    """Save results to CSV file."""
    if not results:
        return
    
    filepath = os.path.join(os.getcwd(), filename)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        
        writer.writerow([
            'stock', 'window_id',
            'train_start', 'train_end', 'test_start', 'test_end',
            'train_return', 'train_sharpe', 'train_max_dd', 'train_win_rate', 
            'train_profit_factor', 'train_trades', 'train_confidence',
            'test_return', 'test_sharpe', 'test_max_dd', 'test_win_rate',
            'test_profit_factor', 'test_trades', 'test_confidence',
            'is_overfitting'
        ])
        
        for r in results:
            writer.writerow([
                r.stock, r.window_id,
                r.train_start, r.train_end, r.test_start, r.test_end,
                f"{r.train_return:.2f}", f"{r.train_sharpe:.2f}", f"{r.train_max_dd:.2f}",
                f"{r.train_win_rate:.2f}", f"{r.train_profit_factor:.2f}", r.train_trades,
                r.train_confidence.value,
                f"{r.test_return:.2f}", f"{r.test_sharpe:.2f}", f"{r.test_max_dd:.2f}",
                f"{r.test_win_rate:.2f}", f"{r.test_profit_factor:.2f}", r.test_trades,
                r.test_confidence.value,
                r.is_overfitting
            ])
    
    print(f"\n[Results saved to: {filepath}]")


def print_validation_report(results: List[WalkForwardResult], summary: ValidationSummary):
    """Print formatted validation report."""
    print("\n" + "=" * 70)
    print("                  WALK-FORWARD VALIDATION REPORT")
    print("=" * 70)
    
    print(f"\n[WINDOW SUMMARY]")
    print(f"   Total Windows:  {summary.total_windows}")
    print(f"   PASS (Robust):  {summary.robust_windows}")
    print(f"   MARGINAL (Weak): {summary.weak_windows}")
    print(f"   FAIL (Overfitted): {summary.overfitted_windows}")
    print(f"   Low Confidence: {summary.low_confidence_windows}")
    
    print(f"\n[AGGREGATED METRICS]")
    print(f"   {'Metric':<20} {'Train':>12} {'Test':>12}")
    print(f"   {'-'*44}")
    print(f"   {'Return (%)':<20} {summary.avg_train_return:>12.2f} {summary.avg_test_return:>12.2f}")
    print(f"   {'Sharpe Ratio':<20} {summary.avg_train_sharpe:>12.2f} {summary.avg_test_sharpe:>12.2f}")
    print(f"   {'Max DD (%)':<20} {summary.avg_train_dd:>12.2f} {summary.avg_test_dd:>12.2f}")
    
    print(f"\n[SAMPLE WINDOWS (first 5)]")
    print(f"   {'Stock':<8} {'Train Return':>14} {'Test Return':>14} {'Train Sharpe':>14} {'Test Sharpe':>14}")
    print(f"   {'-'*70}")
    
    for r in results[:5]:
        print(f"   {r.stock:<8} {r.train_return:>13.1f}% {r.test_return:>13.1f}% "
              f"{r.train_sharpe:>13.2f} {r.test_sharpe:>13.2f}")
    
    if summary.overfitting_flags:
        print(f"\n[OVERFITTING FLAGS]")
        for flag in summary.overfitting_flags[:5]:
            print(f"   - {flag}")
    
    print(f"\n{'='*70}")
    
    if summary.verdict == ValidationVerdict.ROBUST:
        status = "PASS"
    elif summary.verdict == ValidationVerdict.WEAK:
        status = "MARGINAL"
    else:
        status = "FAIL"
    
    print(f"   FINAL VERDICT: {status}")
    print(f"   Reason: {summary.verdict_reason}")
    print(f"{'='*70}\n")


# =============================================================================
# ROBUST BACKTESTER CLASS (ENHANCED)
# =============================================================================

class RobustBacktester:
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
    
    def download_data(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Download multi-year data."""
        stock = yf.Ticker(symbol)
        df = stock.history(start=start, end=end, interval="1d")
        return df
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
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
    
    def run_single_backtest(self, df: pd.DataFrame, allocated_capital: float) -> dict:
        """Run backtest on a dataset."""
        capital = allocated_capital
        trades = []
        equity_curve = [capital]
        position = None
        
        transaction_cost = 0.001
        
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
                    entry = latest['Close'] * (1 + transaction_cost)
                    position = {
                        'type': 'LONG',
                        'entry': entry,
                        'sl': entry * 0.98,
                        'tp': entry * 1.06
                    }
                elif sell_score >= 4 and trend == 'DOWN':
                    entry = latest['Close'] * (1 - transaction_cost)
                    position = {
                        'type': 'SHORT',
                        'entry': entry,
                        'sl': entry * 1.02,
                        'tp': entry * 0.94
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
                    exit_price = price * (1 - transaction_cost) if p['type'] == 'LONG' else price * (1 + transaction_cost)
                    
                    if p['type'] == 'LONG':
                        pnl = (exit_price - p['entry']) * (capital / p['entry'])
                    else:
                        pnl = (p['entry'] - exit_price) * (capital / p['entry'])
                    
                    trades.append({'pnl': pnl, 'exit': exit_reason})
                    capital += pnl
                    equity_curve.append(capital)
                    position = None
        
        if position:
            price = df.iloc[-1]['Close']
            exit_price = price * (1 - transaction_cost) if position['type'] == 'LONG' else price * (1 + transaction_cost)
            if position['type'] == 'LONG':
                pnl = (exit_price - position['entry']) * (capital / position['entry'])
            else:
                pnl = (position['entry'] - exit_price) * (capital / position['entry'])
            trades.append({'pnl': pnl, 'exit': 'END'})
            capital += pnl
            equity_curve.append(capital)
        
        return {
            'trades': trades,
            'equity_curve': equity_curve,
            'final_capital': capital,
            'return_pct': ((capital - allocated_capital) / allocated_capital) * 100
        }
    
    def walk_forward_test(self, symbol: str, train_years: int = 3, 
                        test_years: int = 2) -> dict:
        """Walk-forward testing with enhanced metrics."""
        df = self.download_data(symbol, 
                               start=(datetime.now() - timedelta(days=365*6)).strftime('%Y-%m-%d'),
                               end=datetime.now().strftime('%Y-%m-%d'))
        
        if df is None or len(df) < 1000:
            return {'error': 'Insufficient data'}
        
        df = self.calculate_indicators(df)
        
        split_idx = int(len(df) * (train_years / (train_years + test_years)))
        
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        print(f"\n[{symbol}] Data: {len(df)} days")
        print(f"  Train: {train_df.index[0].date()} to {train_df.index[-1].date()} ({len(train_df)} days)")
        print(f"  Test:  {test_df.index[0].date()} to {test_df.index[-1].date()} ({len(test_df)} days)")
        
        # Train backtest
        train_result = self.run_single_backtest(train_df, self.initial_capital)
        
        # Test backtest (out-of-sample)
        test_result = self.run_single_backtest(test_df, self.initial_capital)
        
        # Calculate ENHANCED metrics using new function
        train_metrics = calculate_enhanced_metrics(
            train_result['trades'], 
            train_result['equity_curve'],
            self.initial_capital
        )
        
        test_metrics = calculate_enhanced_metrics(
            test_result['trades'],
            test_result['equity_curve'],
            self.initial_capital
        )
        
        # Check for overfitting
        is_overfit, reasons = check_overfitting(train_metrics, test_metrics)
        
        return {
            'symbol': symbol,
            'train': train_metrics,
            'test': test_metrics,
            'train_period': f"{train_df.index[0].date()} to {train_df.index[-1].date()}",
            'test_period': f"{test_df.index[0].date()} to {test_df.index[-1].date()}",
            'is_overfitting': is_overfit,
            'overfitting_reasons': reasons
        }


# =============================================================================
# MAIN FUNCTION (ENHANCED)
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("  PHASE 1: ROBUST BACKTESTING (ENHANCED)")
    print("  Walk-Forward Testing with Full Validation")
    print("=" * 70)
    
    # Extended stock list (15 stocks)
    stocks = [
        'GOOGL', 'META', 'AAPL', 'MSFT', 'SPY',  # Top performers
        'AMZN', 'NVDA', 'TSLA', 'NFLX', 'AMD',  # High volatility
        'JPM', 'V', 'MA', 'PYPL', 'SQ'           # Financial/Tech
    ]
    
    tester = RobustBacktester(initial_capital=100000)
    
    results = []
    window_id = 0
    
    print("\n[Running Walk-Forward Tests]")
    print(f"  Stocks: {len(stocks)}")
    print("  Training: 3 years")
    print("  Testing: 2 years (OUT-OF-SAMPLE)")
    print("-" * 70)
    
    for symbol in stocks:
        result = tester.walk_forward_test(symbol, train_years=3, test_years=2)
        if 'error' not in result:
            window_id += 1
            
            # Convert to WalkForwardResult format
            wf_result = WalkForwardResult(
                stock=symbol,
                window_id=window_id,
                train_start=result['train_period'].split(' to ')[0],
                train_end=result['train_period'].split(' to ')[1],
                test_start=result['test_period'].split(' to ')[0],
                test_end=result['test_period'].split(' to ')[1],
                train_return=result['train'].return_pct,
                train_sharpe=result['train'].sharpe_ratio,
                train_max_dd=result['train'].max_drawdown_pct,
                train_win_rate=result['train'].win_rate_pct,
                train_profit_factor=result['train'].profit_factor,
                train_trades=result['train'].total_trades,
                train_confidence=result['train'].confidence,
                test_return=result['test'].return_pct,
                test_sharpe=result['test'].sharpe_ratio,
                test_max_dd=result['test'].max_drawdown_pct,
                test_win_rate=result['test'].win_rate_pct,
                test_profit_factor=result['test'].profit_factor,
                test_trades=result['test'].total_trades,
                test_confidence=result['test'].confidence,
                is_overfitting=result['is_overfitting'],
                overfitting_reasons=result['overfitting_reasons']
            )
            results.append(wf_result)
    
    # Run full validation
    summary = validate_results(results)
    
    # Print validation report
    print_validation_report(results, summary)
    
    # Save to CSV
    save_results_to_csv(results, "phase1_walkforward_results.csv")
    
    # Legacy summary for compatibility
    print("=" * 70)
    print("  WALK-FORWARD RESULTS SUMMARY (LEGACY)")
    print("=" * 70)
    
    print(f"\n{'Symbol':<8} | {'Train Return':>12} | {'Train Sharpe':>12} | {'Test Return':>12} | {'Test Sharpe':>12}")
    print("-" * 70)
    
    for r in results:
        print(f"{r.stock:<8} | {r.train_return:>11.1f}% | {r.train_sharpe:>11.2f} | {r.test_return:>11.1f}% | {r.test_sharpe:>11.2f}")
    
    return results, summary


if __name__ == "__main__":
    results, summary = main()
