"""Legacy: Advanced market regime detector moved to legacy directory.
Contenido movido desde src/system/advanced_market_regime_detector.py
"""

# Copied content (originally in src/system/advanced_market_regime_detector.py)
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

try:
    from .safe_ml_loader import (
        SafeGaussianHMM, get_safe_scaler, get_safe_random_forest,
        is_hmmlearn_available, is_sklearn_available
    )
except Exception:
    try:
        from safe_ml_loader import (
            SafeGaussianHMM, get_safe_scaler, get_safe_random_forest,
            is_hmmlearn_available, is_sklearn_available
        )
    except Exception:
        SafeGaussianHMM = None
        get_safe_scaler = lambda: None
        get_safe_random_forest = lambda: None
        is_hmmlearn_available = lambda: False
        is_sklearn_available = lambda: False

HMM_AVAILABLE = is_hmmlearn_available()
SKLEARN_AVAILABLE = is_sklearn_available()

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    BULL_TRENDING = "bull_trending"
    BEAR_TRENDING = "bear_trending"
    SIDEWAYS_LOW_VOL = "sideways_low_vol"
    SIDEWAYS_HIGH_VOL = "sideways_high_vol"
    BREAKOUT_BULLISH = "breakout_bullish"
    BREAKOUT_BEARISH = "breakout_bearish"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    REVERSAL_BULLISH = "reversal_bullish"
    REVERSAL_BEARISH = "reversal_bearish"

@dataclass
class RegimeFeatures:
    volatility: float
    volume_ratio: float
    momentum: float
    correlation: float
    liquidity_score: float
    trend_strength: float
    mean_reversion: float
    skewness: float
    kurtosis: float
    vix_level: float

@dataclass
class RegimePrediction:
    current_regime: MarketRegime
    regime_probability: float
    transition_probability: float
    next_regime_candidates: List[Tuple[MarketRegime, float]]
    confidence_score: float
    features_importance: Dict[str, float]

class AdvancedMarketRegimeDetector:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logger
        self.n_components = self.config.get('n_components', 6)
        self.covariance_type = self.config.get('covariance_type', 'full')
        self.n_iter = self.config.get('n_iter', 100)
        self.hmm_model = None
        self.scaler = get_safe_scaler()
        self.rf_classifier = get_safe_random_forest()
        self.historical_features = []
        self.historical_regimes = []
        self.feature_history = pd.DataFrame()
        self.state_to_regime = {}
        if HMM_AVAILABLE and SafeGaussianHMM is not None:
            self._initialize_hmm_model()

    def _initialize_hmm_model(self):
        try:
            if SafeGaussianHMM is not None:
                self.hmm_model = SafeGaussianHMM(n_components=self.n_components)
            else:
                self.hmm_model = None
        except Exception:
            self.hmm_model = None

    # (Other methods are present in the original file but omitted here for brevity)
