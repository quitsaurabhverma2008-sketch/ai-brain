"""
Signal Engine Module
====================
Entry/Exit signal generation based on technical indicators
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from enum import Enum


class Signal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SignalGenerator:
    """Generate trading signals based on multiple indicators"""
    
    def __init__(self,
                 rsi_oversold: float = 35,
                 rsi_overbought: float = 65,
                 min_score: int = 3):
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.min_score = min_score
    
    def calculate_buy_score(self, df: pd.DataFrame) -> Tuple[int, dict]:
        """
        Calculate buy score (0-10) based on bullish indicators.
        
        Returns:
            Tuple of (score, details_dict)
        """
        if df.empty or len(df) < 30:
            return 0, {}
        
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        score = 0
        details = {}
        
        # Trend check
        sma20 = latest.get('SMA_20', 0)
        sma50 = latest.get('SMA_50', 0)
        
        trend_up = sma20 > sma50 if (sma20 > 0 and sma50 > 0) else (latest['Close'] > sma20)
        details['trend_up'] = trend_up
        
        # RSI - Oversold (strong buy signal)
        rsi = latest.get('RSI', 50)
        if rsi < self.rsi_oversold:
            score += 3
            details['rsi_oversold'] = True
        elif rsi < 40:
            score += 1
            details['rsi_bullish'] = True
        elif rsi < 45:
            score += 0.5
            details['rsi_near_oversold'] = True
        
        # MACD - Bullish
        macd = latest.get('MACD', 0)
        macd_sig = latest.get('MACD_Signal', 0)
        prev_macd = prev.get('MACD', 0)
        prev_macd_sig = prev.get('MACD_Signal', 0)
        
        if macd > macd_sig:
            score += 1
            details['macd_above'] = True
        
        if prev_macd <= prev_macd_sig and macd > macd_sig:
            score += 2
            details['macd_cross_up'] = True
        
        # Zero line crossover
        if prev_macd < 0 and macd > 0:
            score += 1
            details['macd_zero_cross'] = True
        
        # Price above SMAs
        close = latest['Close']
        if close > sma20 and sma20 > 0:
            score += 1
            details['above_sma20'] = True
        
        if close > sma50 and sma50 > 0:
            score += 1
            details['above_sma50'] = True
        
        # Bollinger Bands - Price near lower band
        bb_lower = latest.get('BB_Lower', 0)
        bb_middle = latest.get('BB_Middle', 0)
        
        if bb_lower > 0 and close <= bb_lower:
            score += 2
            details['at_bb_lower'] = True
        elif bb_middle > 0 and close < bb_middle:
            score += 1
            details['below_bb_middle'] = True
        
        # Price near support (close to lower BB)
        if bb_lower > 0 and close < bb_lower * 1.02:
            score += 1
            details['near_support'] = True
        
        # Stochastic - Oversold
        stoch_k = latest.get('Stoch_K', 50)
        stoch_d = latest.get('Stoch_D', 50)
        if stoch_k < 20:
            score += 2
            details['stoch_oversold'] = True
        elif stoch_k < 30:
            score += 1
            details['stoch_near_oversold'] = True
        
        # Stochastic crossover
        prev_stoch_k = prev.get('Stoch_K', 50)
        prev_stoch_d = prev.get('Stoch_D', 50)
        if prev_stoch_k <= prev_stoch_d and stoch_k > stoch_d:
            score += 1
            details['stoch_cross_up'] = True
        
        # ADX - Strong trend (but not too strong)
        adx = latest.get('ADX', 0)
        plus_di = latest.get('Plus_DI', 0)
        minus_di = latest.get('Minus_DI', 0)
        
        if adx > 20 and plus_di > minus_di:
            score += 1.5
            details['adx_strong_bullish'] = True
        elif adx > 25:
            score += 1
            details['adx_trending'] = True
        
        # CCI - Oversold
        cci = latest.get('CCI', 0)
        if cci < -100:
            score += 2
            details['cci_oversold'] = True
        elif cci < -50:
            score += 1
            details['cci_near_oversold'] = True
        
        # OBV - Rising
        prev_obv = prev.get('OBV', 0)
        obv = latest.get('OBV', 0)
        if obv > prev_obv:
            score += 1
            details['obv_rising'] = True
        
        # VWAP - Price below VWAP (opportunity)
        vwap = latest.get('VWAP', 0)
        if vwap > 0 and close < vwap:
            score += 1
            details['below_vwap'] = True
        
        # Volume confirmation (if available)
        vol = latest.get('Volume', 0)
        vol_sma = latest.get('Volume_SMA', 0)
        if vol > vol_sma and vol > 0:
            score += 0.5
            details['high_volume'] = True
        
        # Strong momentum (RSI rising)
        if len(df) > 5:
            rsi_prev = df.iloc[-5]['RSI']
            if rsi > rsi_prev:
                score += 0.5
                details['rsi_rising'] = True
        
        return score, details
    
    def calculate_sell_score(self, df: pd.DataFrame) -> Tuple[int, dict]:
        """
        Calculate sell score (0-10) based on bearish indicators.
        
        Returns:
            Tuple of (score, details_dict)
        """
        if df.empty:
            return 0, {}
        
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        score = 0
        details = {}
        
        # Trend check
        trend_down = latest['SMA_20'] < latest['SMA_50'] if 'SMA_20' in df.columns else False
        details['trend_down'] = trend_down
        
        # RSI - Overbought (strong sell signal)
        if latest['RSI'] > self.rsi_overbought:
            score += 3
            details['rsi_overbought'] = True
        elif latest['RSI'] > 60:
            score += 1
            details['rsi_bearish'] = True
        
        # MACD - Bearish crossover
        if latest['MACD'] < latest['MACD_Signal']:
            score += 1
            details['macd_below'] = True
        
        if prev['MACD'] >= prev['MACD_Signal'] and latest['MACD'] < latest['MACD_Signal']:
            score += 2
            details['macd_cross_down'] = True
        
        # Price below SMAs
        if latest['Close'] < latest['SMA_20']:
            score += 1
            details['below_sma20'] = True
        
        if latest['Close'] < latest['SMA_50']:
            score += 1
            details['below_sma50'] = True
        
        # Bollinger Bands - Price near upper band (overbought)
        if latest['Close'] >= latest['BB_Upper']:
            score += 2
            details['at_bb_upper'] = True
        elif latest['Close'] > latest['BB_Middle']:
            score += 1
            details['above_bb_middle'] = True
        
        # Stochastic - Overbought
        stoch_k = latest.get('Stoch_K', 50)
        stoch_d = latest.get('Stoch_D', 50)
        if stoch_k > 80:
            score += 2
            details['stoch_overbought'] = True
        elif stoch_k > 70:
            score += 1
            details['stoch_near_overbought'] = True
        
        # Stochastic crossover down
        prev_stoch_k = prev.get('Stoch_K', 50)
        prev_stoch_d = prev.get('Stoch_D', 50)
        if prev_stoch_k >= prev_stoch_d and stoch_k < stoch_d:
            score += 1
            details['stoch_cross_down'] = True
        
        # ADX - Bearish
        adx = latest.get('ADX', 0)
        plus_di = latest.get('Plus_DI', 0)
        minus_di = latest.get('Minus_DI', 0)
        
        if adx > 20 and minus_di > plus_di:
            score += 1.5
            details['adx_strong_bearish'] = True
        
        # CCI - Overbought
        cci = latest.get('CCI', 0)
        if cci > 100:
            score += 2
            details['cci_overbought'] = True
        elif cci > 50:
            score += 1
            details['cci_near_overbought'] = True
        
        # OBV - Falling
        prev_obv = prev.get('OBV', 0)
        obv = latest.get('OBV', 0)
        if obv < prev_obv:
            score += 1
            details['obv_falling'] = True
        
        # VWAP - Price above VWAP
        vwap = latest.get('VWAP', 0)
        if vwap > 0 and latest['Close'] > vwap:
            score += 1
            details['above_vwap'] = True
        
        return score, details
    
    def generate_signal(self, df: pd.DataFrame) -> Tuple[Signal, int, dict]:
        """
        Generate trading signal based on all indicators.
        
        Args:
            df: DataFrame with indicators calculated
        
        Returns:
            Tuple of (Signal, score, details)
        """
        if df.empty or len(df) < 50:
            return Signal.HOLD, 0, {}
        
        buy_score, buy_details = self.calculate_buy_score(df)
        sell_score, sell_details = self.calculate_sell_score(df)
        
        latest = df.iloc[-1]
        
        # Determine trend
        trend = 'UP' if latest['SMA_20'] > latest['SMA_50'] else 'DOWN'
        
        # Combined details
        details = {
            'buy_score': buy_score,
            'sell_score': sell_score,
            'trend': trend,
            'rsi': latest.get('RSI', 50),
            'price': latest['Close'],
            **buy_details,
            **sell_details
        }
        
        # Generate signal
        buy_factors = buy_score >= self.min_score and trend == 'UP'
        sell_factors = sell_score >= self.min_score and trend == 'DOWN'
        
        if buy_factors:
            return Signal.BUY, buy_score, details
        elif sell_factors:
            return Signal.SELL, sell_score, details
        else:
            return Signal.HOLD, max(buy_score, sell_score), details
    
    def get_entry_levels(self, entry_price: float) -> dict:
        """Calculate SL and TP levels."""
        stop_loss = entry_price * 0.97  # 3% stop loss
        take_profit = entry_price * 1.08  # 8% take profit
        risk_reward = 8 / 3  # 2.67 ratio
        
        return {
            'entry_price': entry_price,
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2),
            'risk_percent': 3.0,
            'reward_percent': 8.0,
            'risk_reward_ratio': risk_reward
        }


class ExitSignalGenerator:
    """Generate exit signals for open positions"""
    
    def check_exit(self, df: pd.DataFrame, position_type: str, 
                   entry_price: float, current_price: float) -> Tuple[bool, str]:
        """
        Check if position should be exited.
        
        Args:
            df: DataFrame with indicators
            position_type: 'LONG' or 'SHORT'
            entry_price: Entry price
            current_price: Current price
        
        Returns:
            Tuple of (should_exit, reason)
        """
        if df.empty:
            return False, ""
        
        latest = df.iloc[-1]
        
        if position_type == 'LONG':
            # Check stop loss
            if current_price <= entry_price * 0.98:
                return True, "STOP_LOSS"
            
            # Check take profit
            if current_price >= entry_price * 1.06:
                return True, "TAKE_PROFIT"
            
            # Check RSI overbought
            if latest['RSI'] > 70:
                return True, "RSI_OVERBOUGHT"
            
            # Check MACD bearish crossover
            if len(df) > 1:
                prev = df.iloc[-2]
                if prev['MACD'] >= prev['MACD_Signal'] and latest['MACD'] < latest['MACD_Signal']:
                    return True, "MACD_CROSS_DOWN"
        
        else:  # SHORT
            # Check stop loss
            if current_price >= entry_price * 1.02:
                return True, "STOP_LOSS"
            
            # Check take profit
            if current_price <= entry_price * 0.94:
                return True, "TAKE_PROFIT"
            
            # Check RSI oversold
            if latest['RSI'] < 30:
                return True, "RSI_OVERSOLD"
            
            # Check MACD bullish crossover
            if len(df) > 1:
                prev = df.iloc[-2]
                if prev['MACD'] <= prev['MACD_Signal'] and latest['MACD'] > latest['MACD_Signal']:
                    return True, "MACD_CROSS_UP"
        
        return False, ""


if __name__ == "__main__":
    # Test signal generation
    from trading_brain.data_feed import DataFeed
    from trading_brain.indicators import calculate_all_indicators
    
    feed = DataFeed()
    df = feed.get_intraday_data("GOOGL", "1h", "5d")
    df = calculate_all_indicators(df)
    
    generator = SignalGenerator()
    signal, score, details = generator.generate_signal(df)
    
    print(f"Signal: {signal.value}")
    print(f"Score: {score}")
    print(f"Details: {details}")
