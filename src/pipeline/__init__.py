"""
Paquete de utilidades para orquestación del pipeline en vivo.

Incluye:
- decision_normalizer: normalización robusta de salidas del Decision Agent
- kline_utils: extracción de timestamps y utilidades de deduplicación por vela
"""

from .decision_normalizer import normalize_decision_output  # noqa: F401
from .kline_utils import extract_kline_close_ts, should_process_kline  # noqa: F401
