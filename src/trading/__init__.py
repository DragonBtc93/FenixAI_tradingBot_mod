# src/trading/__init__.py
"""
Fenix Trading Module.

Este módulo contiene el núcleo refactorizado del sistema de trading,
organizado de forma modular y mantenible.

Componentes principales:
- TradingEngine: Motor principal de trading
- OrderExecutor: Ejecutor de órdenes a Binance
- MarketDataManager: Gestión de datos de mercado en tiempo real
- PositionManager: Gestión de posiciones abiertas
"""

from .engine import TradingEngine
from .executor import OrderExecutor
from .market_data import MarketDataManager

__all__ = [
    "TradingEngine",
    "OrderExecutor",
    "MarketDataManager",
]
