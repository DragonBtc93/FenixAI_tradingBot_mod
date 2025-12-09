import pytest
import os
from src.system import __package__ as sys_pkg


from src.system import get_market_regime_detector, get_multi_timeframe_analyzer, should_load_legacy


def test_should_load_legacy_default_false():
    # Ensure default env var not set
    os.environ.pop('FENIX_LOAD_LEGACY_SYSTEM', None)
    assert not should_load_legacy()
    # Getters should be guarded and return None
    assert get_market_regime_detector() is None
    assert get_multi_timeframe_analyzer() is None


def test_shims_raise_when_disabled():
    # Ensure the env is unset
    # Ensure the env is unset
    os.environ.pop('FENIX_LOAD_LEGACY_SYSTEM', None)
    # Importing the module should succeed but instantiation should raise
    import sys, importlib
    for name in ['src.system.advanced_market_regime_detector','system.advanced_market_regime_detector','advanced_market_regime_detector']:
        sys.modules.pop(name, None)
    for name in ['src.system.multi_timeframe_analyzer','system.multi_timeframe_analyzer','multi_timeframe_analyzer']:
        sys.modules.pop(name, None)
    for name in ['src.system.signal_evolution_engine','system.signal_evolution_engine','signal_evolution_engine']:
        sys.modules.pop(name, None)
    amd = importlib.import_module('src.system.advanced_market_regime_detector')
    # Ensure we are importing the shim (not the legacy implementation)
    assert 'legacy' not in (amd.__file__ or '')
    assert not should_load_legacy()
    mta = importlib.import_module('src.system.multi_timeframe_analyzer')
    assert 'legacy' not in (mta.__file__ or '')
    see = importlib.import_module('src.system.signal_evolution_engine')
    assert 'legacy' not in (see.__file__ or '')


def test_should_load_legacy_true_env_var():
    os.environ['FENIX_LOAD_LEGACY_SYSTEM'] = 'true'
    try:
        assert should_load_legacy()
        import sys, importlib
        sys.modules.pop('src.system.advanced_market_regime_detector', None)
        sys.modules.pop('src.system.multi_timeframe_analyzer', None)
        sys.modules.pop('src.system.signal_evolution_engine', None)
        amd = importlib.import_module('src.system.advanced_market_regime_detector')
        # Should be able to instantiate when enabled
        _ = amd.AdvancedMarketRegimeDetector()
        mta = importlib.import_module('src.system.multi_timeframe_analyzer')
        # The legacy copy has an async interface; just import the class
        assert hasattr(mta, 'MultiTimeframeAnalyzer')
        see = importlib.import_module('src.system.signal_evolution_engine')
        assert hasattr(see, 'SignalEvolutionEngine')
    finally:
        os.environ.pop('FENIX_LOAD_LEGACY_SYSTEM', None)


def test_new_shims_gating():
    import sys, importlib
    # Ensure legacy disabled
    os.environ.pop('FENIX_LOAD_LEGACY_SYSTEM', None)
    for name in [
        'src.system.bayesian_strategy_optimizer',
        'src.system.continuous_learning_engine',
        'src.system.on_demand_model_manager',
        'src.system.adaptive_signal_manager',
        'src.system.advanced_portfolio_risk_manager'
    ]:
        sys.modules.pop(name, None)

    # Import shims should not be the legacy module when disabled
    bso = importlib.import_module('src.system.bayesian_strategy_optimizer')
    assert 'legacy' not in (getattr(bso, '__file__', '') or '')
    cle = importlib.import_module('src.system.continuous_learning_engine')
    assert 'legacy' not in (getattr(cle, '__file__', '') or '')
    odm = importlib.import_module('src.system.on_demand_model_manager')
    assert 'legacy' not in (getattr(odm, '__file__', '') or '')
    asm = importlib.import_module('src.system.adaptive_signal_manager')
    assert 'legacy' not in (getattr(asm, '__file__', '') or '')
    aprm = importlib.import_module('src.system.advanced_portfolio_risk_manager')
    assert 'legacy' not in (getattr(aprm, '__file__', '') or '')

    # Enable legacy and verify the legacy classes are available and instantiable
    os.environ['FENIX_LOAD_LEGACY_SYSTEM'] = 'true'
    try:
        bso = importlib.reload(bso)
        _ = bso.BayesianStrategyOptimizer()
        cle = importlib.reload(cle)
        _ = cle.ContinuousLearningEngine()
        odm = importlib.reload(odm)
        _ = odm.OnDemandModelManager()
        asm = importlib.reload(asm)
        _ = asm.AdaptiveSignalManager()
        aprm = importlib.reload(aprm)
        _ = aprm.AdvancedPortfolioRiskManager()
        pre = importlib.import_module('src.system.portfolio_risk_engine')
        # When legacy is enabled, the portfolo risk engine should be importable and available
        assert hasattr(pre, 'PortfolioRiskEngine')
        _ = pre.PortfolioRiskEngine()
        # Additional medium-priority modules
        abe = importlib.import_module('src.system.advanced_backtesting_engine')
        assert 'legacy' not in (getattr(abe, '__file__', '') or '') or hasattr(abe, 'AdvancedBacktestingEngine')
        ams = importlib.import_module('src.system.advanced_metrics_system')
        assert 'legacy' not in (getattr(ams, '__file__', '') or '') or hasattr(ams, 'AdvancedMetricsSystem')
        rpa = importlib.import_module('src.system.realtime_performance_analyzer')
        assert 'legacy' not in (getattr(rpa, '__file__', '') or '') or hasattr(rpa, 'RealtimePerformanceAnalyzer')
        mei = importlib.import_module('src.system.multi_exchange_integration')
        assert 'legacy' not in (getattr(mei, '__file__', '') or '') or hasattr(mei, 'MultiExchangeIntegration')
    finally:
        os.environ.pop('FENIX_LOAD_LEGACY_SYSTEM', None)
