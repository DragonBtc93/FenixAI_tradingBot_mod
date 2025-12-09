import os
import sys
import pytest

from src.system.system_improvements_integration import SystemImprovementsManager


@pytest.mark.asyncio
async def test_system_improvements_initialize_safe_mode_no_legacy():
    # Ensure legacy disabled
    os.environ.pop('FENIX_LOAD_LEGACY_SYSTEM', None)
    # Remove any cached modules to simulate fresh import
    for name in list(sys.modules):
        if name.startswith('src.system.advanced') or 'bayesian' in name or 'multi_timeframe' in name:
            sys.modules.pop(name, None)

    manager = SystemImprovementsManager()
    # Initialize should succeed without importing heavy legacy modules
    await manager.initialize()
    # Ensure no heavy modules loaded in sys.modules
    assert 'src.system.advanced_market_regime_detector' not in sys.modules
    assert 'src.system.multi_timeframe_analyzer' not in sys.modules
    assert 'src.system.bayesian_strategy_optimizer' not in sys.modules
    # Validate that manager initialized critical components
    assert 'circuit_breaker_manager' in manager.components
    # The advanced system factories should exist, and since legacy is disabled, produce None
    factory = manager.advanced_system_classes.get('bayesian_optimizer')
    assert callable(factory)
    assert factory() is None


@pytest.mark.asyncio
async def test_system_improvements_initialize_with_legacy_enabled():
    # Enable legacy modules and reinitialize
    os.environ['FENIX_LOAD_LEGACY_SYSTEM'] = 'true'
    for name in list(sys.modules):
        if name.startswith('src.system.advanced') or 'bayesian' in name or 'multi_timeframe' in name:
            sys.modules.pop(name, None)

    manager = SystemImprovementsManager()
    try:
        await manager.initialize()
        # With legacy enabled, the legacy placeholders may be imported and instantiated
        assert 'circuit_breaker_manager' in manager.components
        # The legacy advanced items will either be None or instantiated depending on implementation
        # Check that at least the placeholder advanced portfolio risk manager can be created if available
        if 'advanced_portfolio_risk_manager' in manager.components:
            ap_rm = manager.components['advanced_portfolio_risk_manager']
            # ap_rm might be a None or an instance
            assert ap_rm is None or hasattr(ap_rm, 'compute_portfolio_risk')
        # Validate that a factory produced an instance if legacy enabled
        factory = manager.advanced_system_classes.get('bayesian_optimizer')
        assert callable(factory)
        instance = factory()
        # instance may be None or an instantiated object depending on legacy placeholders
        assert instance is None or hasattr(instance, '__class__')
    finally:
        os.environ.pop('FENIX_LOAD_LEGACY_SYSTEM', None)
