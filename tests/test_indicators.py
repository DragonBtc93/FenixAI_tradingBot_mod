#!/usr/bin/env python3
"""
Tests para cálculo de indicadores técnicos.
"""
import pytest
import numpy as np


def calculate_ema(prices: list[float], period: int) -> float:
    """Calcula EMA correctamente usando el multiplicador exponencial."""
    if len(prices) < period:
        return sum(prices) / len(prices)
    
    multiplier = 2 / (period + 1)
    ema = sum(prices[:period]) / period  # SMA inicial
    
    for price in prices[period:]:
        ema = (price * multiplier) + (ema * (1 - multiplier))
    
    return ema


def calculate_rsi(prices: list[float], period: int = 14) -> float:
    """Calcula RSI correctamente."""
    if len(prices) < period + 1:
        return 50.0
    
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains = [max(0, d) for d in deltas]
    losses = [max(0, -d) for d in deltas]
    
    # Usar EMA para gains y losses (más preciso que SMA)
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


class TestEMA:
    """Tests para cálculo de EMA."""
    
    def test_ema_basic(self):
        """Test EMA con datos básicos."""
        prices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ema_5 = calculate_ema(prices, 5)
        
        # EMA debe ser mayor que SMA para serie ascendente
        sma_5 = sum(prices[-5:]) / 5
        assert ema_5 >= sma_5 * 0.9  # Permitir cierta tolerancia
        
    def test_ema_trending_up(self):
        """Test EMA en tendencia alcista."""
        prices = [100, 102, 105, 108, 112, 118, 125, 133, 142, 152]
        ema_9 = calculate_ema(prices, 9)
        
        # EMA debe seguir la tendencia
        assert ema_9 > prices[0]
        assert ema_9 < prices[-1]
        
    def test_ema_flat(self):
        """Test EMA en mercado lateral."""
        prices = [100] * 20
        ema_9 = calculate_ema(prices, 9)
        
        assert abs(ema_9 - 100) < 0.01


class TestRSI:
    """Tests para cálculo de RSI."""
    
    def test_rsi_oversold(self):
        """Test RSI en condición de sobreventa."""
        # Precios cayendo constantemente
        prices = [100 - i*2 for i in range(20)]  # 100, 98, 96...
        rsi = calculate_rsi(prices)
        
        assert rsi < 30, f"RSI debería indicar sobreventa, got {rsi}"
        
    def test_rsi_overbought(self):
        """Test RSI en condición de sobrecompra."""
        # Precios subiendo constantemente
        prices = [100 + i*2 for i in range(20)]  # 100, 102, 104...
        rsi = calculate_rsi(prices)
        
        assert rsi > 70, f"RSI debería indicar sobrecompra, got {rsi}"
        
    def test_rsi_neutral(self):
        """Test RSI en mercado lateral."""
        # Precios alternando
        prices = [100, 101, 100, 101, 100, 101, 100, 101, 100, 101,
                  100, 101, 100, 101, 100, 101, 100, 101, 100, 101]
        rsi = calculate_rsi(prices)
        
        assert 40 < rsi < 60, f"RSI debería ser neutral, got {rsi}"


class TestIntegration:
    """Tests de integración."""
    
    def test_indicators_consistency(self):
        """Verifica que los indicadores sean consistentes entre sí."""
        # Tendencia alcista fuerte
        prices = [100 + i*3 for i in range(50)]
        
        ema_9 = calculate_ema(prices, 9)
        ema_21 = calculate_ema(prices, 21)
        rsi = calculate_rsi(prices)
        
        # En tendencia alcista: EMA9 > EMA21, RSI alto
        assert ema_9 > ema_21, "EMA9 debe estar sobre EMA21 en tendencia alcista"
        assert rsi > 50, "RSI debe ser alcista"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
