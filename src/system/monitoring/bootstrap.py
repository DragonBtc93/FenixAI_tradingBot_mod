"""Helpers to start monitoring components without cluttering live_trading."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, Optional

AlertTaskFactory = Callable[[float], Awaitable[None]]


def bootstrap_monitoring_systems(
    logger: logging.Logger,
    monitoring_enabled: bool,
    dashboard: Optional[Any] = None,
    metrics_collector: Optional[Any] = None,
    alert_task_factory: Optional[AlertTaskFactory] = None,
    alert_interval: float = 60.0,
) -> Dict[str, Any]:
    """Start dashboard, metrics loop and alert task if available."""
    results: Dict[str, Any] = {
        "dashboard_started": False,
        "system_monitoring_started": False,
        "alert_task": None,
    }

    if not monitoring_enabled:
        logger.info("Monitoring stack disabled (MONITORING_AVAILABLE=False)")
        return results

    if dashboard is not None:
        try:
            dashboard.start()
            logger.info(
                "🖥️ Dashboard de trading iniciado en http://%s:%s",
                getattr(dashboard, "host", "127.0.0.1"),
                getattr(dashboard, "port", "5000"),
            )
            results["dashboard_started"] = True
        except Exception as exc:
            logger.warning(f"No se pudo iniciar el dashboard de monitoreo: {exc}")

    if metrics_collector is not None:
        try:
            metrics_collector.start_system_monitoring(interval_seconds=30.0)
            logger.info("📈 System monitoring started (30s interval)")
            results["system_monitoring_started"] = True
        except Exception as exc:
            logger.debug(f"No se pudo iniciar el sistema de monitoreo: {exc}")

    if alert_task_factory is not None:
        try:
            results["alert_task"] = asyncio.create_task(
                alert_task_factory(alert_interval)
            )
            logger.info("🔔 Background alert checks task started")
        except Exception as exc:
            logger.debug(f"No se pudo iniciar la tarea de alertas: {exc}")

    return results
