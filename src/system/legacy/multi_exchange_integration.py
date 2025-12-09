"""Legacy MultiExchangeIntegration (simplified placeholder).
This module provides a lightweight placeholder used when legacy modules are enabled.
"""

import logging

logger = logging.getLogger(__name__)

class MultiExchangeIntegration:
    def __init__(self, *args, **kwargs):
        logger.info("Legacy MultiExchangeIntegration initialized (placeholder)")

    def connect_all(self):
        return True

__all__ = ["MultiExchangeIntegration"]
