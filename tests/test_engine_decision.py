import types
import pytest

from src.trading.engine import TradingEngine


class _StubExecutor:
    def __init__(self):
        self.called = False
        self.min_notional = 5.0

    def get_balance(self):
        return 1000.0

    async def execute_market_order(self, _side, quantity, stop_loss=None, take_profit=None):
        self.called = True
        return types.SimpleNamespace(
            success=True,
            status="FILLED",
            executed_qty=quantity,
            entry_price=stop_loss or 1.0,
            message="",
        )


class _StubMarketData:
    def __init__(self, price: float = 100.0):
        self.current_price = price
        self.current_volume = 0.0

    def get_microstructure_metrics(self):
        return types.SimpleNamespace(
            obi=0.0,
            cvd=0.0,
            spread=0.0,
            bid_depth=0.0,
            ask_depth=0.0,
        )


@pytest.mark.asyncio
async def test_process_decision_increments_hold_counter():
    engine = TradingEngine(symbol="BTCUSDT", timeframe="15m", paper_trading=True)
    engine.executor = _StubExecutor()
    engine.market_data = _StubMarketData()

    await engine._process_decision({"final_trade_decision": {"final_decision": "HOLD", "confidence_in_decision": "LOW", "combined_reasoning": "test"}})

    assert engine._consecutive_holds == 1


@pytest.mark.asyncio
async def test_process_decision_executes_paper_trade_without_live_flag():
    engine = TradingEngine(symbol="BTCUSDT", timeframe="15m", paper_trading=True)
    engine.executor = _StubExecutor()
    engine.market_data = _StubMarketData(price=20000)

    await engine._process_decision({"final_trade_decision": {"final_decision": "BUY", "confidence_in_decision": "HIGH", "combined_reasoning": "test", "risk_assessment": {"entry_price": 20000}}})

    # Paper trading short-circuits before executor is called
    assert engine.executor.called is False
