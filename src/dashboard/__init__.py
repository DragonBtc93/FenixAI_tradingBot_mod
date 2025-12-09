# src/dashboard/__init__.py
"""Dashboard module for Fenix Trading System."""
from src.dashboard.trading_dashboard import (
    TradingDashboard,
    LiveDashboard,
    AgentStatus,
    PipelineMetrics,
    get_dashboard,
)

__all__ = [
    "TradingDashboard",
    "LiveDashboard",
    "AgentStatus",
    "PipelineMetrics",
    "get_dashboard",
]
