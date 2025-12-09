"""Legacy Advanced Metrics System (simplified placeholder).
This module provides a lightweight placeholder used when legacy modules are enabled.
"""

import logging

logger = logging.getLogger(__name__)

class AdvancedMetricsSystem:
    def __init__(self, *args, **kwargs):
        logger.info("Legacy AdvancedMetricsSystem initialized (placeholder)")

    def get_report(self):
        return {"uptime_seconds": 0, "cpu_usage_percent": 0.0}

__all__ = ["AdvancedMetricsSystem"]
