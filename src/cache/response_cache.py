"""
Sistema de Cache Avanzado con Redis para FenixAI Trading Bot

Características:
- TTL inteligente por tipo de análisis
- Invalidación automática basada en triggers
- Compresión de datos grandes
- Métricas de performance
- Fallback a cache en memoria si Redis no está disponible
"""

import asyncio
import json
import time
import logging
from typing import Any, Optional, Dict, List, Union
from dataclasses import dataclass
import hashlib

# Imports condicionales para Redis
try:
    from redis import asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    try:
        import aioredis  # Fallback para compatibilidad hacia atrás
        REDIS_AVAILABLE = True
    except ImportError:
        REDIS_AVAILABLE = False
        aioredis = None

from .cache_config import get_cache_config, CacheConfig
from .cache_utils import (
    generate_cache_key,
    is_cache_valid,
    serialize_for_cache,
    deserialize_from_cache,
    calculate_cache_priority
)

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Estadísticas del sistema de cache"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    evictions: int = 0
    errors: int = 0
    total_size_bytes: int = 0
    avg_retrieval_time_ms: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Calcula la tasa de aciertos"""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0


class MemoryCache:
    """Cache en memoria como fallback"""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self.max_size = max_size
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Obtiene valor del cache en memoria"""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    async def set(self, key: str, value: Dict[str, Any], ttl: int = 300) -> bool:
        """Establece valor en cache en memoria"""
        try:
            # Eviction si excede max_size
            if len(self.cache) >= self.max_size:
                await self._evict_lru()
            
            self.cache[key] = value
            self.access_times[key] = time.time()
            return True
        except Exception as e:
            logger.error("Error setting memory cache: %s", e)
            return False
    
    async def delete(self, key: str) -> bool:
        """Elimina clave del cache"""
        if key in self.cache:
            del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
            return True
        return False
    
    async def _evict_lru(self) -> None:
        """Evict usando LRU (Least Recently Used)"""
        if not self.access_times:
            return
        
        # Encontrar clave menos recientemente usada
        lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
        await self.delete(lru_key)
    
    def size(self) -> int:
        """Retorna el tamaño actual del cache"""
        return len(self.cache)


class ResponseCache:
    """
    Sistema de Cache Avanzado con Redis y fallback a memoria
    
    Características:
    - TTL inteligente por tipo de análisis
    - Compresión automática de respuestas grandes
    - Invalidación basada en triggers
    - Métricas detalladas de performance
    - Fallback automático a cache en memoria
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "fenix_cache:",
        compression_threshold: int = 1024,
        enable_compression: bool = True
    ):
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.compression_threshold = compression_threshold
        self.enable_compression = enable_compression
        
        # Redis client
        self.redis: Optional[aioredis.Redis] = None
        self.redis_available = False
        
        # Fallback memory cache
        self.memory_cache = MemoryCache(max_size=500)
        
        # Estadísticas
        self.stats = CacheStats()
        
        # Configuración
        self._retrieval_times: List[float] = []
        
        logger.info("ResponseCache inicializado (Redis: %s)", "enabled" if REDIS_AVAILABLE else "disabled")
    
    async def __aenter__(self):
        """Context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, *args):
        """Context manager exit"""
        await self.close()
    
    async def connect(self) -> None:
        """Conecta a Redis si está disponible"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis no disponible, usando cache en memoria")
            return
        
        try:
            self.redis = aioredis.from_url(self.redis_url)
            # Test connection
            await self.redis.ping()
            self.redis_available = True
            logger.info("✅ Conectado a Redis: %s", self.redis_url)
        except Exception as e:
            logger.warning("❌ Error conectando a Redis: %s. Usando cache en memoria.", e)
            self.redis_available = False
            self.redis = None
    
    async def close(self) -> None:
        """Cierra conexión a Redis"""
        if self.redis:
            await self.redis.close()
            self.redis = None
            self.redis_available = False
    
    def _make_key(self, key: str) -> str:
        """Genera clave completa con prefijo"""
        return f"{self.key_prefix}{key}"
    
    async def get(
        self,
        agent_type: str,
        prompt: str,
        model_id: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Obtiene respuesta del cache
        
        Args:
            agent_type: Tipo de agente
            prompt: Prompt utilizado
            model_id: ID del modelo
            **kwargs: Parámetros adicionales
        
        Returns:
            Respuesta cacheada o None si no existe/expiró
        """
        start_time = time.time()
        
        try:
            # Generar clave de cache
            cache_key = generate_cache_key(agent_type, prompt, model_id, **kwargs)
            full_key = self._make_key(cache_key)
            
            # Obtener configuración de TTL
            config = get_cache_config(agent_type)
            
            # Intentar obtener de Redis primero
            cached_data = None
            if self.redis_available and self.redis:
                try:
                    cached_json = await self.redis.get(full_key)
                    if cached_json:
                        cached_data = json.loads(cached_json)
                except Exception as e:
                    logger.warning("Error obteniendo de Redis: %s", e)
                    self.stats.errors += 1
            
            # Fallback a memoria si Redis falló
            if not cached_data:
                cached_data = await self.memory_cache.get(full_key)
            
            # Verificar validez del cache
            if cached_data and is_cache_valid(cached_data, config.ttl_seconds):
                response = deserialize_from_cache(cached_data)
                
                # Actualizar estadísticas
                self.stats.hits += 1
                retrieval_time = (time.time() - start_time) * 1000
                self._update_avg_retrieval_time(retrieval_time)
                
                logger.debug("🎯 Cache HIT para %s (%.1fms)", agent_type, retrieval_time)
                return response
            
            # Cache miss o expirado
            if cached_data:
                # Cache expirado, eliminar
                await self._delete_key(full_key)
            
            self.stats.misses += 1
            logger.debug("❌ Cache MISS para %s", agent_type)
            return None
            
        except Exception as e:
            logger.error("Error en cache get: %s", e)
            self.stats.errors += 1
            return None
    
    async def set(
        self,
        agent_type: str,
        prompt: str,
        response: str,
        model_id: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Almacena respuesta en cache
        
        Args:
            agent_type: Tipo de agente
            prompt: Prompt utilizado
            response: Respuesta a cachear
            model_id: ID del modelo
            **kwargs: Parámetros adicionales
        
        Returns:
            True si se almacenó correctamente
        """
        try:
            # Generar clave y configuración
            cache_key = generate_cache_key(agent_type, prompt, model_id, **kwargs)
            full_key = self._make_key(cache_key)
            config = get_cache_config(agent_type)
            
            # Serializar datos
            cached_data = serialize_for_cache(response)
            cached_json = json.dumps(cached_data)
            
            # Determinar si comprimir (si está habilitado y supera el threshold)
            should_compress = (
                self.enable_compression and 
                len(cached_json) > self.compression_threshold
            )
            
            if should_compress:
                try:
                    import gzip
                    cached_json = gzip.compress(cached_json.encode()).decode('latin1')
                    cached_data['compressed'] = True
                except ImportError:
                    logger.warning("gzip no disponible, almacenando sin comprimir")
            
            # Intentar almacenar en Redis
            redis_success = False
            if self.redis_available and self.redis:
                try:
                    await self.redis.setex(
                        full_key,
                        config.ttl_seconds,
                        cached_json
                    )
                    redis_success = True
                except Exception as e:
                    logger.warning("Error almacenando en Redis: %s", e)
                    self.stats.errors += 1
            
            # Almacenar en memoria como backup
            memory_success = await self.memory_cache.set(
                full_key, 
                cached_data, 
                config.ttl_seconds
            )
            
            if redis_success or memory_success:
                self.stats.sets += 1
                self.stats.total_size_bytes += len(cached_json)
                logger.debug("✅ Cache SET para %s (%d bytes)", agent_type, len(cached_json))
                return True
            
            return False
            
        except Exception as e:
            logger.error("Error en cache set: %s", e)
            self.stats.errors += 1
            return False
    
    async def invalidate_by_pattern(self, pattern: str) -> int:
        """
        Invalida entradas de cache que coincidan con un patrón
        
        Args:
            pattern: Patrón para buscar claves a invalidar
        
        Returns:
            Número de claves invalidadas
        """
        invalidated = 0
        
        try:
            # Invalidar en Redis
            if self.redis_available and self.redis:
                keys_pattern = f"{self.key_prefix}*{pattern}*"
                keys = await self.redis.keys(keys_pattern)
                if keys:
                    await self.redis.delete(*keys)
                    invalidated += len(keys)
            
            # Invalidar en memoria
            keys_to_delete = [
                key for key in self.memory_cache.cache.keys()
                if pattern in key
            ]
            
            for key in keys_to_delete:
                await self.memory_cache.delete(key)
                invalidated += 1
            
            logger.info("🧹 Invalidadas %d entradas con patrón: %s", invalidated, pattern)
            
        except Exception as e:
            logger.error("Error invalidando cache: %s", e)
            self.stats.errors += 1
        
        return invalidated
    
    async def _delete_key(self, full_key: str) -> None:
        """Elimina una clave específica de ambos caches"""
        try:
            if self.redis_available and self.redis:
                await self.redis.delete(full_key)
            await self.memory_cache.delete(full_key)
        except Exception as e:
            logger.warning("Error eliminando clave %s: %s", full_key, e)
    
    def _update_avg_retrieval_time(self, retrieval_time_ms: float) -> None:
        """Actualiza tiempo promedio de recuperación"""
        self._retrieval_times.append(retrieval_time_ms)
        
        # Mantener solo las últimas 100 mediciones
        if len(self._retrieval_times) > 100:
            self._retrieval_times = self._retrieval_times[-100:]
        
        self.stats.avg_retrieval_time_ms = sum(self._retrieval_times) / len(self._retrieval_times)
    
    def get_stats(self):
        """Obtener estadísticas del cache"""
        return {
            'hits': self.stats.hits,
            'misses': self.stats.misses,
            'hit_rate': self.stats.hit_rate
        }
    
    async def clear_all(self) -> bool:
        """Limpia todo el cache"""
        try:
            if self.redis_available and self.redis:
                pattern = f"{self.key_prefix}*"
                keys = await self.redis.keys(pattern)
                if keys:
                    await self.redis.delete(*keys)
            
            self.memory_cache.cache.clear()
            self.memory_cache.access_times.clear()
            
            logger.info("🧹 Cache completamente limpiado")
            return True
            
        except Exception as e:
            logger.error("Error limpiando cache: %s", e)
            return False


# Instancia global singleton
_global_cache: Optional[ResponseCache] = None


def get_response_cache() -> ResponseCache:
    """Obtiene la instancia global del cache"""
    global _global_cache
    
    if _global_cache is None:
        _global_cache = ResponseCache()
    
    return _global_cache


async def shutdown_cache() -> None:
    """Cierra el cache global"""
    global _global_cache
    
    if _global_cache is not None:
        await _global_cache.close()
        _global_cache = None