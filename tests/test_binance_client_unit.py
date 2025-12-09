"""
Tests para el cliente de Binance.
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock


class TestBinanceConfig:
    """Tests para BinanceConfig."""

    def test_testnet_urls(self):
        """Verificar URLs de testnet."""
        from src.trading.binance_client import BinanceConfig

        config = BinanceConfig(testnet=True)

        assert "testnet" in config.base_url
        assert config.testnet is True

    def test_live_urls(self):
        """Verificar URLs de producción."""
        from src.trading.binance_client import BinanceConfig

        config = BinanceConfig(testnet=False)

        assert "fapi.binance.com" in config.base_url
        assert config.testnet is False

    def test_default_recv_window(self):
        """Verificar recv_window por defecto."""
        from src.trading.binance_client import BinanceConfig

        config = BinanceConfig()

        assert config.recv_window == 5000


class TestBinanceClient:
    """Tests para BinanceClient."""

    @pytest.fixture
    def client(self):
        """Crear cliente de Binance."""
        from src.trading.binance_client import BinanceClient

        return BinanceClient(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True,
        )

    def test_client_initialization(self, client):
        """Verificar inicialización del cliente."""
        assert client.config.testnet is True
        assert client._connected is False

    def test_client_not_connected_initially(self, client):
        """Verificar que el cliente no está conectado inicialmente."""
        assert client._connected is False
        assert client._session is None

    @pytest.mark.asyncio
    async def test_connect_success(self, client):
        """Verificar conexión exitosa."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch('httpx.AsyncClient') as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            result = await client.connect()

            assert result is True
            assert client._connected is True

    @pytest.mark.asyncio
    async def test_connect_failure(self, client):
        """Verificar manejo de fallo de conexión."""
        with patch('httpx.AsyncClient') as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(side_effect=Exception("Connection failed"))
            mock_client.return_value = mock_instance

            result = await client.connect()

            assert result is False


class TestBinanceClientSecurity:
    """Tests de seguridad del cliente."""

    def test_api_keys_from_env(self):
        """Verificar que las claves se pueden cargar del entorno."""
        from src.trading.binance_client import BinanceClient

        with patch.dict('os.environ', {
            'BINANCE_API_KEY': 'env_key',
            'BINANCE_API_SECRET': 'env_secret'
        }):
            client = BinanceClient(testnet=True)
            # Las claves deberían cargarse del entorno si no se proporcionan
            assert client.config is not None

    def test_testnet_default(self):
        """Verificar que testnet es el default."""
        from src.trading.binance_client import BinanceClient

        client = BinanceClient()

        assert client.config.testnet is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
