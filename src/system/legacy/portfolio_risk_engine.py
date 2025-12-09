"""Legacy PortfolioRiskEngine implementation (cleaned).
Copied from the core file, without duplicates.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import statistics
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"

class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"

@dataclass
class PortfolioRiskMetrics:
    var_1d: float
    var_5d: float
    expected_shortfall: float
    max_drawdown: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    beta: float
    correlation_risk: float
    liquidity_risk: float
    concentration_risk: float
    risk_level: RiskLevel

@dataclass
class PositionRisk:
    symbol: str
    position_size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    risk_contribution: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    time_in_position: timedelta
    correlation_with_portfolio: float

@dataclass
class RiskLimits:
    max_position_size: float
    max_portfolio_risk: float
    max_daily_loss: float
    max_drawdown: float
    max_correlation: float
    max_concentration: float
    var_limit: float
    max_leverage: float

class PortfolioRiskEngine:
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, PositionRisk] = {}
        self.price_history: Dict[str, List[float]] = {}
        self.returns_history: Dict[str, List[float]] = {}
        self.portfolio_returns: List[float] = []
        self.risk_limits = RiskLimits(
            max_position_size=0.1,
            max_portfolio_risk=0.02,
            max_daily_loss=0.05,
            max_drawdown=0.15,
            max_correlation=0.7,
            max_concentration=0.3,
            var_limit=0.03,
            max_leverage=2.0
        )
        self.confidence_level = 0.95
        self.lookback_period = 252
        self.stress_scenarios = self._initialize_stress_scenarios()
        self.current_regime = MarketRegime.SIDEWAYS
        self.volatility_regime = "normal"
        self.daily_pnl_history: List[float] = []
        self.max_historical_drawdown = 0.0
        self.consecutive_losses = 0
        self.win_rate = 0.5
        logger.info("PortfolioRiskEngine (legacy) initialized")

    def _initialize_stress_scenarios(self) -> Dict[str, Dict[str, float]]:
        return {
            "market_crash_2008": {"market_drop": -0.30, "volatility_spike": 4.0, "correlation_increase": 0.9},
            "flash_crash": {"market_drop": -0.10, "volatility_spike": 5.0, "liquidity_dry_up": 0.8},
            "crypto_winter": {"market_drop": -0.50, "volatility_spike": 3.0, "correlation_increase": 0.8},
            "high_volatility": {"market_drop": -0.05, "volatility_spike": 2.5, "correlation_increase": 0.6},
            "correlation_breakdown": {"correlation_increase": 0.95, "volatility_spike": 2.0},
            "liquidity_crisis": {"spread_widening": 5.0, "volume_drop": -0.7, "slippage_increase": 3.0},
            "interest_rate_shock": {"market_drop": -0.15, "volatility_spike": 2.0, "sector_rotation": 0.8},
            "geopolitical_crisis": {"market_drop": -0.20, "volatility_spike": 3.5, "safe_haven_flow": 0.9}
        }

    def calculate_portfolio_var(self, confidence_level: float = 0.95) -> float:
        if not self.portfolio_returns or len(self.portfolio_returns) < 20:
            return -0.02
        returns = np.array(self.portfolio_returns[-252:])
        var = np.percentile(returns, (1 - confidence_level) * 100)
        return var

    def calculate_expected_shortfall(self, confidence_level: float = 0.95) -> float:
        var = self.calculate_portfolio_var(confidence_level)
        if not self.portfolio_returns:
            return var * 1.3
        returns = np.array(self.portfolio_returns[-252:])
        tail_returns = returns[returns <= var]
        if len(tail_returns) == 0:
            return var * 1.3
        return np.mean(tail_returns)

    def calculate_portfolio_risk_metrics(self) -> PortfolioRiskMetrics:
        if not self.portfolio_returns or len(self.portfolio_returns) < 20:
            return PortfolioRiskMetrics(var_1d=-0.02, var_5d=-0.04, expected_shortfall=-0.026, max_drawdown=0.0, volatility=0.0, sharpe_ratio=0.0, sortino_ratio=0.0, beta=1.0, correlation_risk=0.0, liquidity_risk=0.0, concentration_risk=0.0, risk_level=RiskLevel.LOW)
        returns = np.array(self.portfolio_returns[-252:])
        var_1d = self.calculate_portfolio_var()
        var_5d = var_1d * np.sqrt(5)
        expected_shortfall = self.calculate_expected_shortfall()
        volatility = np.std(returns) * np.sqrt(252)
        mean_return = np.mean(returns)
        sharpe_ratio = mean_return / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else np.std(returns)
        sortino_ratio = mean_return / downside_std * np.sqrt(252) if downside_std > 0 else 0
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        correlation_risk = 0.0
        liquidity_risk = 0.0
        concentration_risk = 0.0
        risk_level = RiskLevel.LOW
        return PortfolioRiskMetrics(var_1d=var_1d, var_5d=var_5d, expected_shortfall=expected_shortfall, max_drawdown=max_drawdown, volatility=volatility, sharpe_ratio=sharpe_ratio, sortino_ratio=sortino_ratio, beta=1.0, correlation_risk=correlation_risk, liquidity_risk=liquidity_risk, concentration_risk=concentration_risk, risk_level=risk_level)

__all__ = ["PortfolioRiskEngine"] 
