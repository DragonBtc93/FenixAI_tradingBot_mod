"""
Binance Service - Encapsulated Binance client and symbol management
Replaces global binance_client and symbol filter management
"""

import threading
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

try:
    from binance.client import Client
    from binance import enums as binance_enums
    from binance.exceptions import BinanceAPIException, BinanceOrderException
    BINANCE_AVAILABLE = True
except ImportError:
    try:
        from binance import Spot as Client
        from binance import enums as binance_enums
        from binance.exceptions import BinanceAPIException, BinanceOrderException
        BINANCE_AVAILABLE = True
    except ImportError:
        BINANCE_AVAILABLE = False
        Client = None
        binance_enums = None
        
        class BinanceAPIException(Exception):
            pass
        
        class BinanceOrderException(Exception):
            pass

from src.core.trading_constants import get_trading_constants, SymbolConfig

logger = logging.getLogger(__name__)


class BinanceService:
    """
    Encapsulated Binance service that manages client connections
    and symbol configurations without global state
    """
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self._client: Optional[Client] = None
        self._symbol_filters: Dict[str, List[Dict[str, Any]]] = {}
        self._symbol_configs: Dict[str, SymbolConfig] = {}
        self._exchange_info: Optional[Dict[str, Any]] = None
        self._lock = threading.RLock()
        self._initialized = False
        
        if not BINANCE_AVAILABLE:
            logger.warning("Binance client not available. Service will operate in mock mode.")
    
    def initialize(self) -> bool:
        """Initialize Binance client and load exchange info"""
        with self._lock:
            if self._initialized:
                return True
            
            if not BINANCE_AVAILABLE:
                logger.error("Cannot initialize BinanceService: Binance client not available")
                return False
            
            try:
                # Initialize client
                self._client = Client(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    testnet=self.testnet
                )
                
                # Load exchange info
                self._exchange_info = self._client.get_exchange_info()
                
                # Process symbol filters
                self._process_symbol_filters()
                
                self._initialized = True
                logger.info("BinanceService initialized successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to initialize BinanceService: {e}")
                return False
    
    def _process_symbol_filters(self) -> None:
        """Process symbol filters from exchange info"""
        if not self._exchange_info:
            return
        
        for symbol_info in self._exchange_info.get('symbols', []):
            symbol = symbol_info['symbol']
            filters = symbol_info.get('filters', [])
            
            self._symbol_filters[symbol] = filters
            
            # Create or update symbol configuration
            config = SymbolConfig.from_filters(symbol, filters)
            self._symbol_configs[symbol] = config
            
            logger.debug(f"Processed filters for {symbol}")
    
    def is_initialized(self) -> bool:
        """Check if service is initialized"""
        with self._lock:
            return self._initialized
    
    def get_client(self) -> Optional[Client]:
        """Get Binance client instance"""
        return self._client
    
    def get_symbol_filters(self, symbol: str) -> List[Dict[str, Any]]:
        """Get symbol filters"""
        with self._lock:
            return self._symbol_filters.get(symbol, [])
    
    def get_symbol_config(self, symbol: str) -> Optional[SymbolConfig]:
        """Get symbol configuration"""
        with self._lock:
            return self._symbol_configs.get(symbol)
    
    def get_all_symbol_configs(self) -> Dict[str, SymbolConfig]:
        """Get all symbol configurations"""
        with self._lock:
            return self._symbol_configs.copy()
    
    def get_tick_size(self, symbol: str) -> float:
        """Get tick size for symbol"""
        config = self.get_symbol_config(symbol)
        return config.tick_size if config else 0.01
    
    def get_step_size(self, symbol: str) -> float:
        """Get step size for symbol"""
        config = self.get_symbol_config(symbol)
        return config.step_size if config else 0.001
    
    def get_min_notional(self, symbol: str) -> float:
        """Get minimum notional for symbol"""
        config = self.get_symbol_config(symbol)
        return config.min_notional if config else 10.0
    
    def get_price_precision(self, symbol: str) -> int:
        """Get price precision for symbol"""
        config = self.get_symbol_config(symbol)
        return config.price_precision if config else 2
    
    def get_quantity_precision(self, symbol: str) -> int:
        """Get quantity precision for symbol"""
        config = self.get_symbol_config(symbol)
        return config.quantity_precision if config else 3
    
    def format_price(self, symbol: str, price: float) -> str:
        """Format price according to symbol precision"""
        precision = self.get_price_precision(symbol)
        return f"{price:.{precision}f}"
    
    def format_quantity(self, symbol: str, quantity: float) -> str:
        """Format quantity according to symbol precision"""
        precision = self.get_quantity_precision(symbol)
        return f"{quantity:.{precision}f}"
    
    def validate_order(self, symbol: str, quantity: float, price: float) -> Dict[str, Any]:
        """Validate order parameters against symbol filters"""
        config = self.get_symbol_config(symbol)
        if not config:
            return {'valid': False, 'error': f'No configuration for symbol {symbol}'}
        
        errors = []
        
        # Check minimum notional
        notional = quantity * price
        if notional < config.min_notional:
            errors.append(f"Notional {notional} < minimum {config.min_notional}")
        
        # Check price precision
        price_str = str(price)
        if '.' in price_str:
            decimal_places = len(price_str.split('.')[1])
            if decimal_places > config.price_precision:
                errors.append(f"Price precision {decimal_places} > maximum {config.price_precision}")
        
        # Check quantity precision
        quantity_str = str(quantity)
        if '.' in quantity_str:
            decimal_places = len(quantity_str.split('.')[1])
            if decimal_places > config.quantity_precision:
                errors.append(f"Quantity precision {decimal_places} > maximum {config.quantity_precision}")
        
        # Check tick size (price must be multiple of tick size)
        if price % config.tick_size != 0:
            errors.append(f"Price {price} not multiple of tick size {config.tick_size}")
        
        # Check step size (quantity must be multiple of step size)
        if quantity % config.step_size != 0:
            errors.append(f"Quantity {quantity} not multiple of step size {config.step_size}")
        
        if errors:
            return {'valid': False, 'errors': errors}
        
        return {'valid': True}
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get symbol information from exchange info"""
        if not self._exchange_info:
            return None
        
        for symbol_info in self._exchange_info.get('symbols', []):
            if symbol_info['symbol'] == symbol:
                return symbol_info
        
        return None
    
    def get_server_time(self) -> Optional[int]:
        """Get server time from Binance"""
        if not self._client:
            return None
        
        try:
            return self._client.get_server_time()['serverTime']
        except Exception as e:
            logger.error(f"Failed to get server time: {e}")
            return None
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get account information"""
        if not self._client:
            return None
        
        try:
            return self._client.get_account()
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return None
    
    def close(self) -> None:
        """Close Binance client connections"""
        if self._client:
            try:
                self._client.close_connection()
                logger.info("Binance client connection closed")
            except Exception as e:
                logger.error(f"Error closing Binance client: {e}")
            finally:
                self._client = None
                self._initialized = False


# Global service instance
_binance_service: Optional[BinanceService] = None
_service_lock = threading.Lock()


def get_binance_service(api_key: Optional[str] = None, api_secret: Optional[str] = None, testnet: bool = False) -> BinanceService:
    """Get or create the global Binance service instance"""
    global _binance_service
    if _binance_service is None:
        with _service_lock:
            if _binance_service is None:
                _binance_service = BinanceService(api_key, api_secret, testnet)
    return _binance_service


def reset_binance_service() -> None:
    """Reset the global Binance service instance (for testing)"""
    global _binance_service
    with _service_lock:
        if _binance_service:
            _binance_service.close()
        _binance_service = None