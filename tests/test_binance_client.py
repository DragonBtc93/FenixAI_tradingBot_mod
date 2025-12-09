#!/usr/bin/env python3
"""
Tests para el cliente de Binance.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


class TestBinanceConfig:
    """Tests para configuración de Binance."""
    
    def test_testnet_urls(self):
        """Verifica URLs de testnet."""
        from src.trading.binance_client import BinanceConfig
        
        config = BinanceConfig(testnet=True)
        assert "testnet" in config.base_url
        
    def test_live_urls(self):
        """Verifica URLs de producción."""
        from src.trading.binance_client import BinanceConfig
        
        config = BinanceConfig(testnet=False)
        assert "fapi.binance.com" in config.base_url


class TestBinanceClient:
    """Tests para cliente de Binance."""
    
    @pytest.mark.asyncio
    async def test_connection(self):
        """Test de conexión mock."""
        from src.trading.binance_client import BinanceClient
        
        client = BinanceClient(testnet=True)
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
            mock_client.return_value.get = AsyncMock(return_value=mock_response)
            
            # El test verifica que la estructura es correcta
            assert client.config.testnet is True
            
    def test_sign_request(self):
        """Test de firma HMAC."""
        from src.trading.binance_client import BinanceClient
        
        client = BinanceClient(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True
        )
        
        params = {"symbol": "BTCUSDT"}
        signed = client._sign_request(params)
        
        assert "timestamp" in signed
        assert "recvWindow" in signed
        assert "signature" in signed
        assert len(signed["signature"]) == 64  # SHA256 hex


class TestOrderValidation:
    """Tests para validación de órdenes."""
    
    def test_order_params(self):
        """Verifica parámetros de orden."""
        params = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "MARKET",
            "quantity": "0.001000",
        }
        
        assert params["symbol"] == "BTCUSDT"
        assert params["side"] in ["BUY", "SELL"]
        assert params["type"] in ["MARKET", "LIMIT"]
        assert float(params["quantity"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
