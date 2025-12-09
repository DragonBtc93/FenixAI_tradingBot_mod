"""Legacy On-Demand Model Manager (simplified placeholder).
This module provides a lightweight placeholder for the legacy model manager.
"""

import logging

logger = logging.getLogger(__name__)

class OnDemandModelManager:
    def __init__(self, max_cached_models: int = 2):
        self._cache = {}
        self.max_cached = max_cached_models
        logger.info("Legacy OnDemandModelManager initialized (placeholder)")

    def get_model(self, model_id: str):
        # Returns a placeholder model object
        return self._cache.get(model_id)

    def cache_model(self, model_id: str, model_obj):
        if len(self._cache) >= self.max_cached:
            self._cache.pop(next(iter(self._cache)))
        self._cache[model_id] = model_obj

__all__ = ["OnDemandModelManager"]
