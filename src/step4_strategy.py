"""
Step 4: Trading Strategy Module
=================================
"""

import pandas as pd
import numpy as np


class TradingStrategy:
    """
    Trading Strategy - Entry, Exit, SL, TP, Position Sizing
    """
    
    def __init__(self, capital: float = 10000, risk_per_trade: float = 0.02):
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.position = None
    
    def check_buy_signal(self, df: pd.DataFrame) -> dict:
        """Check if BUY signal is generated."""
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        signals = {
            "rsi_oversold": latest['RSI'] < 35,
            "macd_bullish_cross": (latest['MACD'] > latest['MACD_Signal'] and 
                                  prev['MACD'] <= prev['MACD_Signal']),
            "price_above_ma20": latest['Close'] > latest['SMA_20'],
            "price_above_ma50": latest['Close'] > latest['SMA_50'],
            "bb_lower": latest['Close'] < latest['BB_Lower'],
            "bullish_trend": latest['SMA_20'] > latest['SMA_50']
        }
        
        score = sum(signals.values())
        
        return {
            "signal": score >= 3,
            "strength": score,
            "conditions": signals
        }
    
    def check_sell_signal(self, df: pd.DataFrame) -> dict:
        """Check if SELL signal is generated."""
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        signals = {
            "rsi_overbought": latest['RSI'] > 65,
            "macd_bearish_cross": (latest['MACD'] < latest['MACD_Signal'] and 
                                   prev['MACD'] >= prev['MACD_Signal']),
            "price_below_ma20": latest['Close'] < latest['SMA_20'],
            "price_below_ma50": latest['Close'] < latest['SMA_50'],
            "bb_upper": latest['Close'] > latest['BB_Upper'],
            "bearish_trend": latest['SMA_20'] < latest['SMA_50']
        }
        
        score = sum(signals.values())
        
        return {
            "signal": score >= 3,
            "strength": score,
            "conditions": signals
        }
    
    def calculate_sl_tp(self, entry_price: float, direction: str) -> dict:
        """Calculate Stop Loss and Take Profit."""
        sl_percent = 0.02
        tp_percent = 0.06
        
        if direction == "BUY":
            stop_loss = entry_price * (1 - sl_percent)
            take_profit = entry_price * (1 + tp_percent)
        else:
            stop_loss = entry_price * (1 + sl_percent)
            take_profit = entry_price * (1 - tp_percent)
        
        return {
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_percent": sl_percent * 100,
            "reward_percent": tp_percent * 100,
            "risk_reward_ratio": tp_percent / sl_percent
        }
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> dict:
        """Calculate position size."""
        risk_amount = self.capital * self.risk_per_trade
        
        if entry_price > stop_loss:
            price_risk = entry_price - stop_loss
            position_size = risk_amount / price_risk
        else:
            price_risk = stop_loss - entry_price
            position_size = risk_amount / price_risk
        
        capital_required = position_size * entry_price
        
        if capital_required > self.capital:
            position_size = self.capital / entry_price
            actual_risk = position_size * abs(entry_price - stop_loss)
            risk_percent = (actual_risk / self.capital) * 100
        else:
            risk_percent = self.risk_per_trade * 100
        
        return {
            "position_size": position_size,
            "capital_required": capital_required,
            "risk_amount": risk_amount,
            "risk_percent": risk_percent
        }
    
    def analyze(self, df: pd.DataFrame) -> dict:
        """Full analysis - generates complete trade signal."""
        latest = df.iloc[-1]
        
        buy_signal = self.check_buy_signal(df)
        sell_signal = self.check_sell_signal(df)
        
        if buy_signal["signal"]:
            action = "BUY"
            reason = "Multiple bullish conditions met"
        elif sell_signal["signal"]:
            action = "SELL"
            reason = "Multiple bearish conditions met"
        else:
            action = "HOLD"
            reason = "No clear signal"
        
        entry_price = latest['Close']
        
        if action in ["BUY", "SELL"]:
            sl_tp = self.calculate_sl_tp(entry_price, action)
            position = self.calculate_position_size(entry_price, sl_tp["stop_loss"])
            
            return {
                "action": action,
                "entry_price": entry_price,
                "reason": reason,
                "buy_strength": buy_signal["strength"],
                "sell_strength": sell_signal["strength"],
                "stop_loss": sl_tp["stop_loss"],
                "take_profit": sl_tp["take_profit"],
                "risk_reward": sl_tp["risk_reward_ratio"],
                "position_size": position["position_size"],
                "risk_amount": position["risk_amount"],
                "risk_percent": position["risk_percent"]
            }
        else:
            return {
                "action": action,
                "reason": reason,
                "buy_strength": buy_signal["strength"],
                "sell_strength": sell_signal["strength"],
                "indicators": {
                    "rsi": latest['RSI'],
                    "macd": latest['MACD'],
                    "trend": "BEARISH" if latest['SMA_20'] < latest['SMA_50'] else "BULLISH"
                }
            }


def main():
    print("\n" + "=" * 60)
    print("  STEP 4: TRADING STRATEGY")
    print("=" * 60)
    
    print("\n[1] Loading data with indicators...")
    df = pd.read_csv('data/BTC_USD_with_indicators.csv', parse_dates=True, index_col=0)
    print(f"    Loaded {len(df)} rows")
    
    strategy = TradingStrategy(capital=10000, risk_per_trade=0.02)
    
    print("\n[2] Analyzing market...")
    result = strategy.analyze(df)
    
    print("\n" + "=" * 60)
    print("  TRADING SIGNAL")
    print("=" * 60)
    
    if result["action"] in ["BUY", "SELL"]:
        print(f"""
    +---------------------------------------------------------------+
    |  ACTION:          *** {result['action']} ***                           |
    |  Entry Price:     ${result['entry_price']:>10,.2f}                         |
    |  Reason:          {result['reason']:<35}      |
    +---------------------------------------------------------------+
    |  STOP LOSS:       ${result['stop_loss']:>10,.2f}                         |
    |  TAKE PROFIT:     ${result['take_profit']:>10,.2f}                         |
    |  Risk:Reward:     1:{result['risk_reward']:.1f}                                       |
    +---------------------------------------------------------------+
    |  Position Size:   {result['position_size']:.4f} units                      |
    |  Risk Amount:     ${result['risk_amount']:>10,.2f}                         |
    |  Risk %:          {result['risk_percent']:.2f}%                                      |
    +---------------------------------------------------------------+
        """)
    else:
        print(f"""
    +---------------------------------------------------------------+
    |  ACTION:          *** {result['action']} ***                           |
    |  Reason:          {result['reason']:<35}      |
    +---------------------------------------------------------------+
    |  Signal Strength:                                              |
    |    Buy Conditions:  {result['buy_strength']}/6                              |
    |    Sell Conditions: {result['sell_strength']}/6                              |
    +---------------------------------------------------------------+
    |  Current Indicators:                                          |
    |    RSI: {result['indicators']['rsi']:.2f}                                             |
    |    MACD: {result['indicators']['macd']:.2f}                                          |
    |    Trend: {result['indicators']['trend']:<10}                                    |
    +---------------------------------------------------------------+
        """)
    
    print("\n[3] Strategy rules explanation:")
    print("""
    +---------------------------------------------------------------+
    |  BUY CONDITIONS (need 3+):                                    |
    |    - RSI < 35 (oversold)                                      |
    |    - MACD bullish crossover                                   |
    |    - Price above SMA-20                                       |
    |    - Price above SMA-50                                       |
    |    - Price below lower Bollinger Band                         |
    |    - Bullish trend (SMA-20 > SMA-50)                          |
    +---------------------------------------------------------------+
    |  SELL CONDITIONS (need 3+):                                   |
    |    - RSI > 65 (overbought)                                   |
    |    - MACD bearish crossover                                   |
    |    - Price below SMA-20                                       |
    |    - Price below SMA-50                                       |
    |    - Price above upper Bollinger Band                         |
    |    - Bearish trend (SMA-20 < SMA-50)                          |
    +---------------------------------------------------------------+
    |  RISK MANAGEMENT:                                            |
    |    - Max 2% risk per trade                                    |
    |    - 2% Stop Loss                                             |
    |    - 6% Take Profit (3:1 ratio)                               |
    +---------------------------------------------------------------+
    """)
    
    return result


if __name__ == "__main__":
    main()