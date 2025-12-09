"""
Configuración de Cache para diferentes tipos de análisis
"""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class CacheConfig:
    """Configuración de cache por tipo de análisis"""
    ttl_seconds: int
    max_entries: int = 1000
    invalidation_patterns: list = None
    priority: int = 1  # 1=highest


# Configuraciones de TTL optimizadas por tipo de análisis
CACHE_CONFIGS: Dict[str, CacheConfig] = {
    # Sentiment Analysis - Cambia frecuentemente
    'sentiment': CacheConfig(
        ttl_seconds=300,  # 5 minutos
        max_entries=500,
        invalidation_patterns=['news', 'social'],
        priority=2
    ),
    
    # Technical Analysis - Estable por períodos cortos
    'technical': CacheConfig(
        ttl_seconds=180,  # 3 minutos
        max_entries=800,
        invalidation_patterns=['price', 'volume'],
        priority=1
    ),
    
    # Visual Analysis - Más costoso, cache más tiempo
    'visual': CacheConfig(
        ttl_seconds=600,  # 10 minutos
        max_entries=200,
        invalidation_patterns=['chart', 'image'],
        priority=1
    ),
    
    # QABBA Analysis - Cálculos matemáticos, cache más tiempo
    'qabba': CacheConfig(
        ttl_seconds=240,  # 4 minutos
        max_entries=600,
        invalidation_patterns=['price', 'indicators'],
        priority=1
    ),
    
    # Decision Analysis - Combina otros análisis, TTL corto
    'decision': CacheConfig(
        ttl_seconds=120,  # 2 minutos
        max_entries=300,
        invalidation_patterns=['all'],
        priority=3
    ),
    
    # Model Responses - Cache general para respuestas de modelos
    'model_response': CacheConfig(
        ttl_seconds=300,  # 5 minutos
        max_entries=1000,
        priority=2
    )
}


def get_cache_config(analysis_type: str) -> CacheConfig:
    """Obtiene configuración de cache para un tipo de análisis"""
    return CACHE_CONFIGS.get(analysis_type, CACHE_CONFIGS['model_response'])