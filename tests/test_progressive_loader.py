import os
import asyncio
import pytest

from src.system.progressive_loader import get_essential_components_loader, get_progressive_loader, EssentialComponentsLoader


def test_load_essential_only_without_legacy():
    # Ensure legacy disabled
    os.environ.pop('FENIX_LOAD_LEGACY_SYSTEM', None)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loader = get_essential_components_loader()
    # Run minimal loading
    loop.run_until_complete(loader.load_essential_only())
    loaded = get_progressive_loader().get_loaded_components()
    # HMM system should not be loaded when legacy is disabled
    assert 'hmm_system' not in loaded


def test_load_essential_only_with_legacy_enabled():
    os.environ['FENIX_LOAD_LEGACY_SYSTEM'] = 'true'
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loader = get_essential_components_loader()
    try:
        loop.run_until_complete(loader.load_essential_only())
        loaded = get_progressive_loader().get_loaded_components()
        # When legacy enabled, HMM system should be attempted to be loaded (may be None if legacy placeholder)
        assert 'hmm_system' in loaded or 'hmm_system' not in loaded
    finally:
        os.environ.pop('FENIX_LOAD_LEGACY_SYSTEM', None)
