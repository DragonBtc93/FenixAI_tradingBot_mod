import os
import importlib
import sys

import pytest

from src.system import get_portfolio_risk_engine, should_load_legacy


def test_portfolio_risk_engine_disabled_by_default():
    os.environ.pop('FENIX_LOAD_LEGACY_SYSTEM', None)
    assert not should_load_legacy()
    kern = get_portfolio_risk_engine()
    assert kern is None


def test_portfolio_risk_engine_enabled_with_env():
    os.environ['FENIX_LOAD_LEGACY_SYSTEM'] = 'true'
    try:
        kern = get_portfolio_risk_engine()
        # Should return a class or factory; instantiate it
        assert kern is not None
        obj = kern()
        assert hasattr(obj, 'calculate_portfolio_risk_metrics') or hasattr(obj, 'calculate_portfolio_var')
    finally:
        os.environ.pop('FENIX_LOAD_LEGACY_SYSTEM', None)
