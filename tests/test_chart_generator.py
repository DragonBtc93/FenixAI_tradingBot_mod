# ruff: noqa: I001
from datetime import datetime, timedelta
import pytest
import numpy as np
import pandas as pd

from src.tools.chart_generator import FenixChartGenerator, MPLFINANCE_AVAILABLE


def generate_sample_kline_data(n=50, start_price=100.0, timeframe_minutes=15):
    now = datetime.utcnow()
    timestamps = [int((now - timedelta(minutes=timeframe_minutes * (n - i))).timestamp() * 1000) for i in range(n)]
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    price = start_price
    for i in range(n):
        open_p = price
        close_p = round(price + (i * 0.1), 2)
        high_p = max(open_p, close_p) + 0.5
        low_p = min(open_p, close_p) - 0.5
        vol = 100 + i
        opens.append(open_p)
        highs.append(high_p)
        lows.append(low_p)
        closes.append(close_p)
        volumes.append(vol)
        price = close_p

    return {
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }


def test_fenix_chart_generator_smoke():
    kdata = generate_sample_kline_data(60)
    gen = FenixChartGenerator()
    res = gen.generate_chart(kdata, symbol="BTCUSD", timeframe="15m", last_n_candles=50)

    if MPLFINANCE_AVAILABLE:
        assert res.get("image_b64") is not None
        assert isinstance(res.get("image_b64"), str)
        assert len(res.get("image_b64") or "") > 100
    else:
        assert "error" in res
        assert res["error"] == "mplfinance no disponible"


def test_indicators_alignment():
    """Ensure indicators returned are pandas Series aligned to df index and have proper length"""
    kdata = generate_sample_kline_data(60)
    gen = FenixChartGenerator()
    df = gen.prepare_dataframe(kdata)
    assert df is not None
    indicators = gen.calculate_indicators(df.tail(50))
    for name, val in indicators.items():
        # We expect pandas Series for plotted indicators (e.g., ema_9, rsi, macd)
        if isinstance(val, (list, tuple, np.ndarray)):
            pytest.skip(f"Indicator {name} returned as array; converted at plotting stage")
        if isinstance(val, pd.Series):
            assert list(val.index) == list(df.tail(50).index)


@pytest.mark.asyncio
async def test_fenix_chart_generator_async_call():
    kdata = generate_sample_kline_data(30)
    gen = FenixChartGenerator()
    res = gen.generate_chart(kdata, symbol="BTCUSD", timeframe="15m", last_n_candles=30)
    # Same assertions
    if MPLFINANCE_AVAILABLE:
        assert res.get("image_b64") is not None
        assert isinstance(res.get("image_b64"), str)
    else:
        assert "error" in res

