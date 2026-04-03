"""
ML Trading Model
================
XGBoost based prediction model for stock direction
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Using RandomForest instead.")
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


FEATURE_COLS = [
    'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'SMA_20', 'SMA_50', 'EMA_20',
    'BB_Upper', 'BB_Middle', 'BB_Lower',
    'Stoch_K', 'Stoch_D', 'ADX', 'Plus_DI', 'Minus_DI',
    'CCI', 'ATR', 'VWAP', 'Volume_SMA'
]

TARGET_COL = 'target'


class MLTradingModel:
    """
    Machine Learning model for stock direction prediction.
    Predicts whether price will go UP (1) or DOWN (0) in next period.
    """
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # For price regression
        self.regressor = None
        self.regressor_scaler = StandardScaler()
        self.regressor_trained = False
        self.feature_cols = FEATURE_COLS
        
    def create_dataset(self, data_dict: Dict[str, pd.DataFrame], 
                       lookback: int = 24, target: int = 1) -> pd.DataFrame:
        """
        Create training dataset from multiple stock dataframes.
        
        Args:
            data_dict: Dictionary of symbol -> DataFrame
            lookback: Hours to look back for features
            target: Hours ahead to predict
        
        Returns:
            DataFrame with features and target
        """
        from .indicators import calculate_all_indicators
        
        all_data = []
        
        for symbol, df in data_dict.items():
            if df is None or len(df) < lookback + target + 50:
                continue
            
            try:
                df = calculate_all_indicators(df.copy())
            except Exception as e:
                print(f"Error calculating indicators for {symbol}: {e}")
                continue
            
            df = df.dropna()
            
            if len(df) < lookback + target:
                continue
            
            for i in range(lookback, len(df) - target):
                row = {'symbol': symbol}
                
                future_price = df['Close'].iloc[i + target]
                current_price = df['Close'].iloc[i]
                row['target'] = 1 if future_price > current_price else 0
                
                for col in self.feature_cols:
                    if col in df.columns:
                        row[col] = df[col].iloc[i]
                
                if row.get('target') is not None:
                    all_data.append(row)
        
        return pd.DataFrame(all_data)
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target for training"""
        
        feature_df = df[self.feature_cols].copy()
        
        feature_df = feature_df.fillna(0)
        feature_df = feature_df.replace([np.inf, -np.inf], 0)
        
        X = feature_df.values
        y = df[TARGET_COL].values
        
        X = self.scaler.fit_transform(X)
        
        return X, y
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2,
              save: bool = True) -> Dict:
        """
        Train the model.
        
        Args:
            df: Training data
            test_size: Fraction for test set
            save: Whether to save model
        
        Returns:
            Dictionary with training metrics
        """
        if len(df) < 100:
            print("Not enough data for training")
            return {'accuracy': 0, 'samples': 0}
        
        X, y = self.prepare_features(df)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=True
        )
        
        if XGBOOST_AVAILABLE:
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.is_trained = True
        
        metrics = {
            'accuracy': accuracy,
            'samples': len(df),
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        if save:
            self.save_model()
        
        print(f"\nModel trained successfully!")
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  Samples: {len(df)}")
        
        return metrics
    
    def predict_proba(self, features: np.ndarray) -> Tuple[float, float]:
        """
        Predict probability of UP or DOWN.
        
        Args:
            features: Feature array
        
        Returns:
            Tuple of (prob_down, prob_up)
        """
        if self.model is None:
            return 0.5, 0.5
        
        try:
            features = features.reshape(1, -1)
            features = self.scaler.transform(features)
            
            proba = self.model.predict_proba(features)[0]
            
            return proba[0], proba[1]
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0.5, 0.5
    
    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Predict price direction.
        
        Args:
            features: Feature array
        
        Returns:
            Tuple of (direction, confidence)
        """
        prob_down, prob_up = self.predict_proba(features)
        
        if prob_up > 0.5:
            return "UP", prob_up
        elif prob_down > 0.5:
            return "DOWN", prob_down
        else:
            return "HOLD", max(prob_up, prob_down)
    
    def analyze_current(self, df: pd.DataFrame) -> Dict:
        """
        Analyze current market situation.
        
        Args:
            df: DataFrame with latest data and indicators
        
        Returns:
            Dictionary with prediction results
        """
        if df is None or len(df) < 24:
            return {
                'direction': 'HOLD',
                'confidence': 0,
                'prob_up': 0.5,
                'prob_down': 0.5,
                'message': 'Insufficient data'
            }
        
        latest = df.iloc[-1]
        
        feature_values = []
        for col in self.feature_cols:
            val = latest.get(col, 0)
            if pd.isna(val) or np.isinf(val):
                val = 0
            feature_values.append(val)
        
        direction, confidence = self.predict(np.array(feature_values))
        prob_down, prob_up = self.predict_proba(np.array(feature_values))
        
        signal_map = {
            'UP': 'BUY',
            'DOWN': 'SELL',
            'HOLD': 'HOLD'
        }
        
        return {
            'direction': signal_map.get(direction, 'HOLD'),
            'confidence': confidence * 100,
            'prob_up': prob_up * 100,
            'prob_down': prob_down * 100,
            'ml_signal': signal_map.get(direction, 'HOLD')
        }
    
    def prepare_regression_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for price regression"""
        from .indicators import calculate_all_indicators
        
        df = df.copy()
        df = calculate_all_indicators(df)
        df = df.dropna()
        
        if len(df) < 50:
            return np.array([]), np.array([])
        
        feature_df = df[self.feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        
        X = feature_df.values
        y = df['Close'].values
        
        return X, y
    
    def predict_future_prices(self, df: pd.DataFrame, steps: int = 4) -> Dict:
        """
        Predict future prices for N steps.
        Balanced prediction with mean reversion, volatility scaling, and market structure.
        """
        if df is None or len(df) < 50:
            return {
                'prices': [],
                'avg_change': 0,
                'confidence': 0,
                'direction': 'HOLD',
                'scenarios': {}
            }
        
        try:
            latest = df.iloc[-1]
            current_price = latest['Close']
            
            # Get technical indicators
            rsi = latest.get('RSI', 50)
            macd = latest.get('MACD', 0)
            macd_signal = latest.get('MACD_Signal', 0)
            sma_20 = latest.get('SMA_20', current_price)
            sma_50 = latest.get('SMA_50', current_price)
            atr = latest.get('ATR', current_price * 0.02)
            volume = latest.get('Volume', 0)
            volume_sma = latest.get('Volume_SMA', volume or 1)
            
            # Check volume spike
            volume_ratio = volume / volume_sma if volume_sma > 0 else 1
            high_volume_mode = volume_ratio > 2.0
            
            # Calculate current price deviation from SMAs
            deviation_from_sma20 = (current_price - sma_20) / sma_20
            deviation_from_sma50 = (current_price - sma_50) / sma_50
            
            # Determine trend from SMA crossover
            trend = 'UP' if sma_20 > sma_50 else 'DOWN'
            
            # Calculate volatility from recent price action
            recent_prices = df['Close'].tail(20)
            volatility = recent_prices.std() / recent_prices.mean()
            
            # Use ATR-based volatility (more accurate)
            atr_volatility = atr / current_price
            
            # Get ML classifier prediction
            feature_values = []
            for col in self.feature_cols:
                val = latest.get(col, 0)
                if pd.isna(val) or np.isinf(val):
                    val = 0
                feature_values.append(val)
            
            features = np.array(feature_values).reshape(1, -1)
            features = self.scaler.transform(features)
            proba = self.model.predict_proba(features)[0]
            prob_up = proba[1]
            prob_down = proba[0]
            
            # Technical analysis signals
            tech_bullish = sum([
                rsi < 40,
                macd > macd_signal,
                trend == 'UP'
            ])
            
            tech_bearish = sum([
                rsi > 60,
                macd < macd_signal,
                trend == 'DOWN'
            ])
            
            # Calculate base direction
            if tech_bullish > tech_bearish + 1:
                base_direction = 1  # UP
                confidence = min((prob_up + 0.25) * 100, 80)
            elif tech_bearish > tech_bullish + 1:
                base_direction = -1  # DOWN
                confidence = min((prob_down + 0.25) * 100, 80)
            else:
                if prob_up > prob_down:
                    base_direction = 1
                    confidence = prob_up * 100
                else:
                    base_direction = -1
                    confidence = prob_down * 100
            
            # Check for overbought RSI deceleration
            if rsi > 70:
                confidence *= 0.7
                base_direction *= 0.5  # Reduce momentum
            elif rsi > 60:
                confidence *= 0.85
                base_direction *= 0.75
            
            # Generate predicted prices with improved logic
            predicted_prices = []
            cumulative_pct = 0
            
            # Maximum allowed deviation (15% in 2 days unless high volume)
            max_deviation = 0.15 if high_volume_mode else 0.10
            
            for step in range(steps):
                # Time decay factor - predictions become less certain
                time_decay = 0.85 ** step
                
                # Dynamic confidence band - widens with time (funnel effect)
                # Use ATR for realistic uncertainty
                step_uncertainty = atr_volatility * (1 + step * 0.3) * time_decay
                
                # Stochastic noise - market moves in waves
                wave_phase = (step % 3) / 3 * 2 * np.pi
                stochastic_noise = volatility * 0.3 * np.sin(wave_phase)
                
                # Mean reversion - pull back if too far from SMA
                mean_reversion = 0
                if abs(deviation_from_sma20) > 0.05:  # >5% deviation
                    # Pull back toward SMA
                    mean_reversion = -deviation_from_sma20 * 0.1 * time_decay
                
                # Base momentum with time decay
                base_momentum = base_direction * atr_volatility * time_decay
                
                # Volume-weighted adjustment
                if high_volume_mode and step == 0:
                    base_momentum *= 1.5  # Stronger first move on high volume
                
                # Combine all factors
                step_change = (base_momentum + mean_reversion + stochastic_noise) * (1 - step * 0.08)
                
                # Apply max deviation constraint
                cumulative_pct += step_change
                cumulative_pct = max(-max_deviation, min(max_deviation, cumulative_pct))
                
                # Calculate next price
                next_price = current_price * (1 + cumulative_pct)
                predicted_prices.append(round(next_price, 2))
            
            # Calculate final metrics
            final_change = ((predicted_prices[-1] - current_price) / current_price) * 100 if predicted_prices else 0
            
            # Generate confidence bands (wider with time - funnel)
            confidence_bands = []
            for i, price in enumerate(predicted_prices):
                band_width = atr_volatility * (1 + i * 0.4) * current_price
                confidence_bands.append({
                    'upper': round(price + band_width, 2),
                    'lower': round(price - band_width, 2)
                })
            
            return {
                'prices': predicted_prices,
                'current_price': current_price,
                'avg_change': final_change,
                'confidence': confidence,
                'direction': 'UP' if base_direction > 0 else 'DOWN',
                'prob_up': prob_up * 100,
                'prob_down': prob_down * 100,
                'volatility': volatility * 100,
                'atr_volatility': atr_volatility * 100,
                'tech_bullish': tech_bullish,
                'tech_bearish': tech_bearish,
                'trend': trend,
                'rsi': rsi,
                'volume_ratio': volume_ratio,
                'high_volume_mode': high_volume_mode,
                'deviation_sma20': deviation_from_sma20 * 100,
                'confidence_bands': confidence_bands,
                'mean_reversion': abs(deviation_from_sma20) > 0.05
            }
            
        except Exception as e:
            print(f"Price prediction error: {e}")
            return {
                'prices': [],
                'avg_change': 0,
                'confidence': 0,
                'direction': 'HOLD',
                'error': str(e)
            }
    
    def save_model(self, name: str = "trading_model") -> bool:
        """Save model to disk"""
        if self.model is None:
            return False
        
        try:
            model_path = os.path.join(self.model_dir, f"{name}.pkl")
            scaler_path = os.path.join(self.model_dir, f"{name}_scaler.pkl")
            
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            print(f"Model saved to {model_path}")
            return True
            
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, name: str = "trading_model") -> bool:
        """Load model from disk"""
        try:
            model_path = os.path.join(self.model_dir, f"{name}.pkl")
            scaler_path = os.path.join(self.model_dir, f"{name}_scaler.pkl")
            
            if not os.path.exists(model_path):
                print(f"Model not found: {model_path}")
                return False
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            self.is_trained = True
            print(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores"""
        if self.model is None:
            return pd.DataFrame()
        
        if XGBOOST_AVAILABLE:
            importance = self.model.feature_importances_
        else:
            importance = self.model.feature_importances_
        
        return pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)


def train_model_from_data():
    """Train model using collected data"""
    from .data_collector import DataCollector
    
    print("Loading data...")
    collector = DataCollector("data")
    
    data_dict = {}
    
    from data_feed import DataFeed
    feed = DataFeed()
    
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'SPY']
    
    for symbol in stocks:
        print(f"Downloading {symbol}...")
        df = feed.get_intraday_data(symbol, "1h", "730d")
        if df is not None:
            data_dict[symbol] = df
    
    print(f"Downloaded {len(data_dict)} symbols")
    
    print("\nCreating training dataset...")
    model = MLTradingModel()
    
    train_df = model.create_dataset(data_dict, lookback=24, target=1)
    print(f"Training samples: {len(train_df)}")
    
    if len(train_df) < 100:
        print("Not enough data. Need at least 100 samples.")
        return None
    
    metrics = model.train(train_df)
    
    return model


if __name__ == "__main__":
    model = train_model_from_data()
    
    if model:
        importance = model.get_feature_importance()
        print("\nTop 10 Important Features:")
        print(importance.head(10))
