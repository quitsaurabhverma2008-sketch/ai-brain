"""
Trader Module - Position & Risk Management
==========================================
Portfolio tracking, position management, risk controls
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class PositionType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class ExitReason(Enum):
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    MANUAL = "MANUAL"
    SIGNAL_CHANGE = "SIGNAL_CHANGE"
    RSI_EXIT = "RSI_EXIT"
    MACD_EXIT = "MACD_EXIT"


@dataclass
class Position:
    """Represents a single trading position"""
    stock: str
    entry_date: datetime
    entry_price: float
    quantity: int
    position_type: PositionType
    stop_loss: float
    take_profit: float
    current_price: float = 0.0
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    notes: str = ""
    
    @property
    def pnl(self) -> float:
        """Calculate profit/loss for the position"""
        if self.exit_price is None:
            return 0.0
        
        if self.position_type == PositionType.LONG:
            return (self.exit_price - self.entry_price) * self.quantity
        else:  # SHORT
            return (self.entry_price - self.exit_price) * self.quantity
    
    @property
    def pnl_percent(self) -> float:
        """Calculate profit/loss percentage"""
        if self.entry_price == 0:
            return 0.0
        
        if self.position_type == PositionType.LONG:
            return ((self.exit_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - self.exit_price) / self.entry_price) * 100
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L at current price"""
        if self.current_price == 0 or self.quantity == 0:
            return 0.0
        
        if self.position_type == PositionType.LONG:
            return (self.current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.current_price) * self.quantity


@dataclass
class Trade:
    """Completed trade record"""
    stock: str
    entry_date: datetime
    entry_price: float
    exit_date: datetime
    exit_price: float
    position_type: str
    quantity: int
    pnl: float
    pnl_percent: float
    exit_reason: str
    holding_period_hours: float = 0.0


class Portfolio:
    """Manages trading portfolio and positions"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Trade] = []
        self.max_position_size_pct = 0.10  # Max 10% per position
        self.max_daily_loss_pct = 0.05  # Max 5% daily loss
        self.risk_per_trade_pct = 0.02  # 2% risk per trade
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> int:
        """Calculate position size based on risk."""
        risk_amount = self.initial_capital * self.risk_per_trade_pct
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share > 0.01:  # Avoid division by zero
            shares = int(risk_amount / risk_per_share)
        else:
            # Default to 10% of capital if no valid SL
            shares = int((self.initial_capital * 0.10) / entry_price)
        
        # Apply position size limit
        max_shares = int((self.initial_capital * self.max_position_size_pct) / entry_price)
        
        return max(1, min(shares, max_shares))
    
    def open_position(self, stock: str, entry_price: float, 
                    position_type: PositionType, stop_loss: float,
                    take_profit: float) -> bool:
        """
        Open a new position.
        
        Args:
            stock: Stock symbol
            entry_price: Entry price
            position_type: LONG or SHORT
            stop_loss: Stop loss price
            take_profit: Take profit price
        
        Returns:
            True if position opened successfully
        """
        # Check if already have position
        if stock in self.positions:
            return False
        
        # Calculate position size
        quantity = self.calculate_position_size(entry_price, stop_loss)
        
        if quantity <= 0:
            return False
        
        # Check capital
        required_capital = entry_price * quantity
        
        if required_capital > self.cash:
            return False
        
        # Create position
        position = Position(
            stock=stock,
            entry_date=datetime.now(),
            entry_price=entry_price,
            quantity=quantity,
            position_type=position_type,
            stop_loss=stop_loss,
            take_profit=take_profit,
            current_price=entry_price
        )
        
        self.positions[stock] = position
        self.cash -= required_capital
        
        return True
    
    def close_position(self, stock: str, exit_price: float, 
                     reason: ExitReason) -> Optional[Trade]:
        """
        Close an existing position.
        
        Args:
            stock: Stock symbol
            exit_price: Exit price
            reason: Reason for exit
        
        Returns:
            Trade record if position closed
        """
        if stock not in self.positions:
            return None
        
        position = self.positions[stock]
        
        position.exit_date = datetime.now()
        position.exit_price = exit_price
        position.exit_reason = reason.value
        
        # Calculate P&L
        pnl = position.pnl
        pnl_percent = position.pnl_percent
        
        # Calculate holding period
        holding_hours = (position.exit_date - position.entry_date).total_seconds() / 3600
        
        # Create trade record
        trade = Trade(
            stock=stock,
            entry_date=position.entry_date,
            entry_price=position.entry_price,
            exit_date=position.exit_date,
            exit_price=exit_price,
            position_type=position.position_type.value,
            quantity=position.quantity,
            pnl=pnl,
            pnl_percent=pnl_percent,
            exit_reason=reason.value,
            holding_period_hours=holding_hours
        )
        
        # Update cash
        self.cash += (exit_price * position.quantity)
        
        # Remove position
        del self.positions[stock]
        
        # Add to history
        self.trade_history.append(trade)
        
        return trade
    
    def update_prices(self, prices: Dict[str, float]):
        """
        Update current prices for all positions.
        
        Args:
            prices: Dictionary of stock -> current price
        """
        for stock, position in self.positions.items():
            if stock in prices:
                position.current_price = prices[stock]
    
    def check_exits(self, prices: Dict[str, float], exit_triggered: Dict[str, str]) -> List[Trade]:
        """
        Check if any positions should be exited.
        
        Args:
            prices: Current prices
            exit_triggered: Dict of stock -> exit reason
        
        Returns:
            List of closed trades
        """
        self.update_prices(prices)
        
        closed_trades = []
        
        for stock, reason in exit_triggered.items():
            if stock in self.positions:
                position = self.positions[stock]
                trade = self.close_position(stock, position.current_price, ExitReason(reason))
                if trade:
                    closed_trades.append(trade)
        
        return closed_trades
    
    def get_unrealized_pnl(self) -> float:
        """Get total unrealized P&L."""
        return sum(p.unrealized_pnl for p in self.positions.values())
    
    def get_total_value(self) -> float:
        """Get total portfolio value."""
        position_value = sum(p.current_price * p.quantity for p in self.positions.values())
        return self.cash + position_value
    
    def get_stats(self) -> dict:
        """Get portfolio statistics."""
        closed_trades = [t for t in self.trade_history if t.exit_date is not None]
        
        if not closed_trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0
            }
        
        wins = [t for t in closed_trades if t.pnl > 0]
        losses = [t for t in closed_trades if t.pnl < 0]
        
        total_pnl = sum(t.pnl for t in closed_trades)
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        
        return {
            'total_trades': len(closed_trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': (len(wins) / len(closed_trades)) * 100,
            'total_pnl': total_pnl,
            'avg_win': sum(t.pnl for t in wins) / len(wins) if wins else 0,
            'avg_loss': sum(t.pnl for t in losses) / len(losses) if losses else 0,
            'profit_factor': gross_profit / gross_loss if gross_loss > 0 else 0,
            'current_capital': self.get_total_value()
        }


if __name__ == "__main__":
    # Test portfolio
    portfolio = Portfolio(100000)
    
    print(f"Initial Capital: ${portfolio.initial_capital}")
    
    # Test opening position
    opened = portfolio.open_position(
        stock="GOOGL",
        entry_price=295.0,
        position_type=PositionType.LONG,
        stop_loss=289.0,
        take_profit=312.0
    )
    
    print(f"Position Opened: {opened}")
    print(f"Cash: ${portfolio.cash}")
    print(f"Positions: {list(portfolio.positions.keys())}")
