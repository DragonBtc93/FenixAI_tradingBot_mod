from src.api import server


def test_build_system_metrics_populates_history():
    before = len(server._METRICS_HISTORY)
    metrics = server.build_system_metrics()

    assert "cpu" in metrics and "memory" in metrics and "disk" in metrics
    assert metrics["cpu"].get("usage") is not None
    # history grows until maxlen
    assert len(server._METRICS_HISTORY) >= min(before, server._METRICS_HISTORY.maxlen or before)
