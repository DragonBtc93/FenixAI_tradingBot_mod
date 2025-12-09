"""
Hybrid Metrics System para FenixAI Trading Bot
Sistema de métricas avanzado para monitorear performance MLX vs HuggingFace por agente
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import threading
from enum import Enum

logger = logging.getLogger(__name__)


class ModelBackend(Enum):
    """Backends de modelos disponibles"""
    MLX = "mlx"
    HUGGINGFACE = "huggingface"
    HYBRID = "hybrid"


class AgentType(Enum):
    """Tipos de agentes del sistema"""
    SENTIMENT = "sentiment"
    TECHNICAL = "technical"
    VISUAL = "visual"
    QABBA = "qabba"
    DECISION = "decision"
    RISK = "risk"


@dataclass
class MetricSnapshot:
    """Snapshot de métricas en un momento específico"""
    timestamp: datetime
    backend: ModelBackend
    agent_type: AgentType
    response_time_ms: float
    success: bool
    tokens_input: int = 0
    tokens_output: int = 0
    cost_usd: float = 0.0
    model_id: str = ""
    error_type: Optional[str] = None
    cache_hit: bool = False
    rate_limited: bool = False


@dataclass
class AgentMetrics:
    """Métricas agregadas por agente"""
    agent_type: AgentType
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time_ms: float = 0.0
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    cache_hit_rate: float = 0.0
    rate_limit_hits: int = 0
    
    # Métricas por backend
    mlx_requests: int = 0
    mlx_success_rate: float = 0.0
    mlx_avg_time_ms: float = 0.0
    
    hf_requests: int = 0
    hf_success_rate: float = 0.0
    hf_avg_time_ms: float = 0.0
    
    # Métricas de tiempo
    last_request_time: Optional[datetime] = None
    requests_last_hour: int = 0
    requests_last_day: int = 0
    
    @property
    def success_rate(self) -> float:
        """Tasa de éxito general"""
        return (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0.0
    
    @property
    def performance_score(self) -> float:
        """Score de performance (0-100) basado en éxito y velocidad"""
        if self.total_requests == 0:
            return 0.0
        
        # Factor de éxito (60% del score)
        success_factor = self.success_rate * 0.6
        
        # Factor de velocidad (40% del score)
        # Velocidad ideal: < 1000ms = 100%, > 5000ms = 0%
        speed_score = max(0, 100 - (self.avg_response_time_ms - 1000) / 40)
        speed_factor = speed_score * 0.4
        
        return min(100, success_factor + speed_factor)


@dataclass
class SystemMetrics:
    """Métricas del sistema completo"""
    uptime_seconds: float = 0.0
    total_requests: int = 0
    total_successful: int = 0
    total_cost_usd: float = 0.0
    
    # Distribución por backend
    mlx_percentage: float = 0.0
    hf_percentage: float = 0.0
    
    # Métricas de eficiencia
    cache_efficiency: float = 0.0
    rate_limit_efficiency: float = 0.0
    
    # Alertas activas
    active_alerts: List[str] = field(default_factory=list)
    
    @property
    def overall_success_rate(self) -> float:
        """Tasa de éxito general del sistema"""
        return (self.total_successful / self.total_requests * 100) if self.total_requests > 0 else 0.0


class HybridMetricsCollector:
    """
    Colector de métricas híbridas para el sistema FenixAI
    
    Características:
    - Métricas en tiempo real por agente y backend
    - Dashboard de performance comparativo MLX vs HF
    - Alertas automáticas por degradación de performance
    - Análisis de costos y eficiencia
    - Recomendaciones de optimización automáticas
    """
    
    def __init__(
        self,
        retention_hours: int = 24,
        alert_thresholds: Optional[Dict[str, float]] = None,
        enable_dashboard: bool = True
    ):
        self.retention_hours = retention_hours
        self.enable_dashboard = enable_dashboard
        self.start_time = datetime.now()
        
        # Configuración de alertas
        self.alert_thresholds = alert_thresholds or {
            'success_rate_min': 85.0,  # Mínimo 85% éxito
            'response_time_max': 5000.0,  # Máximo 5s respuesta
            'cost_budget_daily': 10.0,  # Máximo $10/día
            'rate_limit_max': 0.1  # Máximo 10% rate limited
        }
        
        # Storage de métricas
        self._snapshots: deque = deque(maxlen=10000)  # Últimas 10k métricas
        self._agent_metrics: Dict[AgentType, AgentMetrics] = {}
        self._system_metrics = SystemMetrics()
        
        # Threading para procesamiento en background
        self._lock = threading.RLock()
        self._dashboard_running = False
        self._dashboard_task = None
        
        # Inicializar métricas de agentes
        for agent_type in AgentType:
            self._agent_metrics[agent_type] = AgentMetrics(agent_type=agent_type)
        
        logger.info("HybridMetricsCollector inicializado con retención de %dh", retention_hours)
    
    def record_metric(
        self,
        backend: ModelBackend,
        agent_type: AgentType,
        response_time_ms: float,
        success: bool,
        tokens_input: int = 0,
        tokens_output: int = 0,
        cost_usd: float = 0.0,
        model_id: str = "",
        error_type: Optional[str] = None,
        cache_hit: bool = False,
        rate_limited: bool = False
    ) -> None:
        """Registra una nueva métrica"""
        
        snapshot = MetricSnapshot(
            timestamp=datetime.now(),
            backend=backend,
            agent_type=agent_type,
            response_time_ms=response_time_ms,
            success=success,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost_usd=cost_usd,
            model_id=model_id,
            error_type=error_type,
            cache_hit=cache_hit,
            rate_limited=rate_limited
        )
        
        with self._lock:
            self._snapshots.append(snapshot)
            self._update_agent_metrics(snapshot)
            self._update_system_metrics(snapshot)
            self._check_alerts()
    
    def _update_agent_metrics(self, snapshot: MetricSnapshot) -> None:
        """Actualiza métricas del agente"""
        metrics = self._agent_metrics[snapshot.agent_type]
        
        # Contadores básicos
        metrics.total_requests += 1
        if snapshot.success:
            metrics.successful_requests += 1
        else:
            metrics.failed_requests += 1
        
        # Tiempo de respuesta promedio
        total_time = metrics.avg_response_time_ms * (metrics.total_requests - 1)
        metrics.avg_response_time_ms = (total_time + snapshot.response_time_ms) / metrics.total_requests
        
        # Costos y tokens
        metrics.total_cost_usd += snapshot.cost_usd
        metrics.total_tokens += snapshot.tokens_input + snapshot.tokens_output
        
        # Rate limiting
        if snapshot.rate_limited:
            metrics.rate_limit_hits += 1
        
        # Métricas por backend
        if snapshot.backend == ModelBackend.MLX:
            metrics.mlx_requests += 1
            mlx_success = sum(1 for s in self._snapshots 
                            if s.agent_type == snapshot.agent_type 
                            and s.backend == ModelBackend.MLX 
                            and s.success)
            metrics.mlx_success_rate = (mlx_success / metrics.mlx_requests * 100)
            
            # Tiempo promedio MLX
            mlx_times = [s.response_time_ms for s in self._snapshots 
                        if s.agent_type == snapshot.agent_type 
                        and s.backend == ModelBackend.MLX]
            metrics.mlx_avg_time_ms = sum(mlx_times) / len(mlx_times) if mlx_times else 0
            
        elif snapshot.backend == ModelBackend.HUGGINGFACE:
            metrics.hf_requests += 1
            hf_success = sum(1 for s in self._snapshots 
                           if s.agent_type == snapshot.agent_type 
                           and s.backend == ModelBackend.HUGGINGFACE 
                           and s.success)
            metrics.hf_success_rate = (hf_success / metrics.hf_requests * 100)
            
            # Tiempo promedio HF
            hf_times = [s.response_time_ms for s in self._snapshots 
                       if s.agent_type == snapshot.agent_type 
                       and s.backend == ModelBackend.HUGGINGFACE]
            metrics.hf_avg_time_ms = sum(hf_times) / len(hf_times) if hf_times else 0
        
        # Cache hit rate
        cache_hits = sum(1 for s in self._snapshots 
                        if s.agent_type == snapshot.agent_type and s.cache_hit)
        metrics.cache_hit_rate = (cache_hits / metrics.total_requests * 100)
        
        # Timestamps
        metrics.last_request_time = snapshot.timestamp
        
        # Requests por período
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        
        metrics.requests_last_hour = sum(1 for s in self._snapshots 
                                       if s.agent_type == snapshot.agent_type 
                                       and s.timestamp > hour_ago)
        metrics.requests_last_day = sum(1 for s in self._snapshots 
                                      if s.agent_type == snapshot.agent_type 
                                      and s.timestamp > day_ago)
    
    def _update_system_metrics(self, snapshot: MetricSnapshot) -> None:
        """Actualiza métricas del sistema"""
        self._system_metrics.uptime_seconds = (datetime.now() - self.start_time).total_seconds()
        self._system_metrics.total_requests = len(self._snapshots)
        self._system_metrics.total_successful = sum(1 for s in self._snapshots if s.success)
        self._system_metrics.total_cost_usd = sum(s.cost_usd for s in self._snapshots)
        
        # Distribución por backend
        mlx_count = sum(1 for s in self._snapshots if s.backend == ModelBackend.MLX)
        hf_count = sum(1 for s in self._snapshots if s.backend == ModelBackend.HUGGINGFACE)
        total = mlx_count + hf_count
        
        if total > 0:
            self._system_metrics.mlx_percentage = (mlx_count / total * 100)
            self._system_metrics.hf_percentage = (hf_count / total * 100)
        
        # Eficiencia
        cache_hits = sum(1 for s in self._snapshots if s.cache_hit)
        rate_limited = sum(1 for s in self._snapshots if s.rate_limited)
        
        self._system_metrics.cache_efficiency = (cache_hits / len(self._snapshots) * 100) if self._snapshots else 0
        self._system_metrics.rate_limit_efficiency = 100 - (rate_limited / len(self._snapshots) * 100) if self._snapshots else 100
    
    def _check_alerts(self) -> None:
        """Verifica y genera alertas automáticas"""
        alerts = []
        
        # Alert por tasa de éxito baja
        if self._system_metrics.overall_success_rate < self.alert_thresholds['success_rate_min']:
            alerts.append(f"⚠️ Tasa de éxito baja: {self._system_metrics.overall_success_rate:.1f}%")
        
        # Alert por tiempo de respuesta alto
        recent_times = [s.response_time_ms for s in list(self._snapshots)[-100:]]  # Últimas 100
        if recent_times:
            avg_time = sum(recent_times) / len(recent_times)
            if avg_time > self.alert_thresholds['response_time_max']:
                alerts.append(f"⚠️ Tiempo respuesta alto: {avg_time:.0f}ms")
        
        # Alert por costo excesivo
        daily_cost = sum(s.cost_usd for s in self._snapshots 
                        if s.timestamp > datetime.now() - timedelta(days=1))
        if daily_cost > self.alert_thresholds['cost_budget_daily']:
            alerts.append(f"⚠️ Presupuesto diario excedido: ${daily_cost:.2f}")
        
        # Alert por rate limiting excesivo
        rate_limited_rate = self._system_metrics.rate_limit_efficiency
        if rate_limited_rate < (100 - self.alert_thresholds['rate_limit_max'] * 100):
            alerts.append(f"⚠️ Rate limiting excesivo: {100-rate_limited_rate:.1f}%")
        
        self._system_metrics.active_alerts = alerts
        
        # Log alerts nuevas
        for alert in alerts:
            if alert not in getattr(self, '_last_alerts', []):
                logger.warning("ALERT: %s", alert)
        
        self._last_alerts = alerts
    
    def get_agent_metrics(self, agent_type: AgentType) -> AgentMetrics:
        """Obtiene métricas de un agente específico"""
        with self._lock:
            return self._agent_metrics.get(agent_type, AgentMetrics(agent_type=agent_type))
    
    def get_system_metrics(self) -> SystemMetrics:
        """Obtiene métricas del sistema"""
        with self._lock:
            return self._system_metrics
    
    def get_comparison_report(self) -> Dict[str, Any]:
        """Genera reporte comparativo MLX vs HuggingFace"""
        with self._lock:
            report = {
                'timestamp': datetime.now().isoformat(),
                'system_overview': {
                    'total_requests': self._system_metrics.total_requests,
                    'success_rate': self._system_metrics.overall_success_rate,
                    'uptime_hours': self._system_metrics.uptime_seconds / 3600,
                    'total_cost': self._system_metrics.total_cost_usd,
                    'mlx_usage': self._system_metrics.mlx_percentage,
                    'hf_usage': self._system_metrics.hf_percentage
                },
                'backend_comparison': {},
                'agent_performance': {},
                'recommendations': self._generate_recommendations()
            }
            
            # Comparación por backend
            mlx_snapshots = [s for s in self._snapshots if s.backend == ModelBackend.MLX]
            hf_snapshots = [s for s in self._snapshots if s.backend == ModelBackend.HUGGINGFACE]
            
            report['backend_comparison'] = {
                'mlx': {
                    'requests': len(mlx_snapshots),
                    'success_rate': (sum(1 for s in mlx_snapshots if s.success) / len(mlx_snapshots) * 100) if mlx_snapshots else 0,
                    'avg_response_time': sum(s.response_time_ms for s in mlx_snapshots) / len(mlx_snapshots) if mlx_snapshots else 0,
                    'total_cost': sum(s.cost_usd for s in mlx_snapshots)
                },
                'huggingface': {
                    'requests': len(hf_snapshots),
                    'success_rate': (sum(1 for s in hf_snapshots if s.success) / len(hf_snapshots) * 100) if hf_snapshots else 0,
                    'avg_response_time': sum(s.response_time_ms for s in hf_snapshots) / len(hf_snapshots) if hf_snapshots else 0,
                    'total_cost': sum(s.cost_usd for s in hf_snapshots)
                }
            }
            
            # Performance por agente
            for agent_type in AgentType:
                metrics = self._agent_metrics[agent_type]
                report['agent_performance'][agent_type.value] = {
                    'total_requests': metrics.total_requests,
                    'success_rate': metrics.success_rate,
                    'performance_score': metrics.performance_score,
                    'avg_response_time': metrics.avg_response_time_ms,
                    'cost': metrics.total_cost_usd,
                    'mlx_performance': {
                        'requests': metrics.mlx_requests,
                        'success_rate': metrics.mlx_success_rate,
                        'avg_time': metrics.mlx_avg_time_ms
                    },
                    'hf_performance': {
                        'requests': metrics.hf_requests,
                        'success_rate': metrics.hf_success_rate,
                        'avg_time': metrics.hf_avg_time_ms
                    }
                }
            
            return report
    
    def _generate_recommendations(self) -> List[str]:
        """Genera recomendaciones automáticas de optimización"""
        recommendations = []
        
        # Analizar performance por backend
        mlx_snapshots = [s for s in self._snapshots if s.backend == ModelBackend.MLX]
        hf_snapshots = [s for s in self._snapshots if s.backend == ModelBackend.HUGGINGFACE]
        
        if mlx_snapshots and hf_snapshots:
            mlx_success = sum(1 for s in mlx_snapshots if s.success) / len(mlx_snapshots)
            hf_success = sum(1 for s in hf_snapshots if s.success) / len(hf_snapshots)
            
            mlx_time = sum(s.response_time_ms for s in mlx_snapshots) / len(mlx_snapshots)
            hf_time = sum(s.response_time_ms for s in hf_snapshots) / len(hf_snapshots)
            
            # Recomendaciones basadas en performance
            if mlx_success > hf_success + 0.1:  # MLX 10% mejor
                recommendations.append("🚀 MLX muestra mejor tasa de éxito. Considerar priorizar MLX para tareas críticas.")
            elif hf_success > mlx_success + 0.1:  # HF 10% mejor
                recommendations.append("☁️ HuggingFace muestra mejor tasa de éxito. Considerar aumentar uso de HF.")
            
            if mlx_time < hf_time * 0.7:  # MLX 30% más rápido
                recommendations.append("⚡ MLX es significativamente más rápido. Priorizar para respuestas rápidas.")
            elif hf_time < mlx_time * 0.7:  # HF 30% más rápido
                recommendations.append("🌐 HuggingFace es más rápido. Optimizar ruteo hacia HF.")
        
        # Recomendaciones por agente
        for agent_type in AgentType:
            metrics = self._agent_metrics[agent_type]
            
            if metrics.performance_score < 50 and metrics.total_requests > 10:
                recommendations.append(f"⚠️ {agent_type.value}: Performance baja ({metrics.performance_score:.1f}). Revisar configuración.")
            
            if metrics.cache_hit_rate < 20 and metrics.total_requests > 50:
                recommendations.append(f"💾 {agent_type.value}: Cache hit rate bajo ({metrics.cache_hit_rate:.1f}%). Revisar estrategia de cache.")
        
        # Recomendaciones de costo
        if self._system_metrics.total_cost_usd > 5.0:  # Más de $5
            recommendations.append("💰 Costos altos detectados. Considerar optimizar uso de modelos premium.")
        
        return recommendations
    
    async def start_dashboard(self, update_interval: int = 30) -> None:
        """Inicia dashboard en tiempo real (para desarrollo)"""
        if not self.enable_dashboard or self._dashboard_running:
            return
        
        self._dashboard_running = True
        logger.info("Dashboard de métricas iniciado (intervalo: %ds)", update_interval)
        
        try:
            while self._dashboard_running:
                self._print_dashboard()
                await asyncio.sleep(update_interval)
        except asyncio.CancelledError:
            pass
        finally:
            self._dashboard_running = False
            logger.info("Dashboard de métricas detenido")
    
    def _print_dashboard(self) -> None:
        """Imprime dashboard en consola (para desarrollo)"""
        print("\n" + "="*80)
        print("🔥 FENIXAI HYBRID METRICS DASHBOARD")
        print("="*80)
        
        # Sistema general
        print(f"⏰ Uptime: {self._system_metrics.uptime_seconds/3600:.1f}h | "
              f"📊 Requests: {self._system_metrics.total_requests} | "
              f"✅ Success: {self._system_metrics.overall_success_rate:.1f}% | "
              f"💰 Cost: ${self._system_metrics.total_cost_usd:.2f}")
        
        print(f"🖥️  MLX: {self._system_metrics.mlx_percentage:.1f}% | "
              f"☁️ HF: {self._system_metrics.hf_percentage:.1f}% | "
              f"💾 Cache: {self._system_metrics.cache_efficiency:.1f}%")
        
        # Por agente
        print("\n📈 PERFORMANCE POR AGENTE:")
        for agent_type in AgentType:
            metrics = self._agent_metrics[agent_type]
            if metrics.total_requests > 0:
                print(f"  {agent_type.value:10} | "
                      f"Req: {metrics.total_requests:3} | "
                      f"Success: {metrics.success_rate:5.1f}% | "
                      f"Time: {metrics.avg_response_time_ms:6.0f}ms | "
                      f"Score: {metrics.performance_score:5.1f}")
        
        # Alertas
        if self._system_metrics.active_alerts:
            print("\n🚨 ALERTAS ACTIVAS:")
            for alert in self._system_metrics.active_alerts:
                print(f"  {alert}")
        
        print("="*80)
    
    def stop_dashboard(self) -> None:
        """Detiene el dashboard"""
        self._dashboard_running = False
    
    def export_metrics(self, format: str = "json") -> str:
        """Exporta métricas en formato especificado"""
        report = self.get_comparison_report()
        
        if format.lower() == "json":
            return json.dumps(report, indent=2, default=str)
        else:
            raise ValueError(f"Formato no soportado: {format}")
    
    def cleanup_old_metrics(self) -> None:
        """Limpia métricas antiguas según retención configurada"""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        
        with self._lock:
            # Mantener solo snapshots recientes
            recent_snapshots = [s for s in self._snapshots if s.timestamp > cutoff_time]
            self._snapshots.clear()
            self._snapshots.extend(recent_snapshots)
            
            logger.info("Limpieza de métricas: mantenidas %d de las últimas %dh", 
                       len(recent_snapshots), self.retention_hours)
    
    async def record_inference_request(
        self,
        agent_type: str,
        provider: str,
        model_id: str,
        latency: float,
        success: bool,
        cost: float,
        response_time_ms: int,
        **kwargs
    ) -> None:
        """
        Método de compatibilidad para registrar requests de inferencia
        
        Args:
            agent_type: Tipo de agente (sentiment, technical, etc.)
            provider: Proveedor (mlx, huggingface, hybrid)
            model_id: ID del modelo utilizado
            latency: Latencia en segundos
            success: Si el request fue exitoso
            cost: Costo del request en USD
            response_time_ms: Tiempo de respuesta en milisegundos
        """
        try:
            # Convertir strings a enums
            backend = ModelBackend.MLX if provider.lower() == "mlx" else ModelBackend.HUGGINGFACE
            if provider.lower() == "hybrid":
                backend = ModelBackend.HYBRID
                
            # Mapear agent_type string a enum
            agent_enum_map = {
                'sentiment': AgentType.SENTIMENT,
                'technical': AgentType.TECHNICAL,
                'visual': AgentType.VISUAL,
                'qabba': AgentType.QABBA,
                'decision': AgentType.DECISION,
                'risk': AgentType.RISK
            }
            
            agent_enum = agent_enum_map.get(agent_type.lower(), AgentType.SENTIMENT)
            
            # Registrar métrica usando el método existente
            self.record_metric(
                backend=backend,
                agent_type=agent_enum,
                response_time_ms=response_time_ms,
                success=success,
                cost_usd=cost,
                model_id=model_id,
                tokens_input=kwargs.get('tokens_input', 100),
                tokens_output=kwargs.get('tokens_output', 50)
            )
            
        except Exception as e:
            logger.error("Error registrando métrica de inference: %s", e)
    
    async def get_provider_comparison(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtiene comparación entre proveedores
        
        Returns:
            Dict con estadísticas por proveedor
        """
        try:
            with self._lock:
                mlx_snapshots = [s for s in self._snapshots if s.backend == ModelBackend.MLX]
                hf_snapshots = [s for s in self._snapshots if s.backend == ModelBackend.HUGGINGFACE]
                
                comparison = {}
                
                # Estadísticas MLX
                if mlx_snapshots:
                    mlx_success = sum(1 for s in mlx_snapshots if s.success)
                    mlx_total = len(mlx_snapshots)
                    mlx_avg_time = sum(s.response_time_ms for s in mlx_snapshots) / mlx_total
                    mlx_total_cost = sum(s.cost_usd for s in mlx_snapshots)
                    
                    comparison['mlx'] = {
                        'request_count': mlx_total,
                        'success_rate': (mlx_success / mlx_total * 100) if mlx_total > 0 else 0,
                        'avg_response_time': mlx_avg_time,
                        'total_cost': mlx_total_cost
                    }
                else:
                    comparison['mlx'] = {
                        'request_count': 0,
                        'success_rate': 0,
                        'avg_response_time': 0,
                        'total_cost': 0
                    }
                
                # Estadísticas HuggingFace
                if hf_snapshots:
                    hf_success = sum(1 for s in hf_snapshots if s.success)
                    hf_total = len(hf_snapshots)
                    hf_avg_time = sum(s.response_time_ms for s in hf_snapshots) / hf_total
                    hf_total_cost = sum(s.cost_usd for s in hf_snapshots)
                    
                    comparison['huggingface'] = {
                        'request_count': hf_total,
                        'success_rate': (hf_success / hf_total * 100) if hf_total > 0 else 0,
                        'avg_response_time': hf_avg_time,
                        'total_cost': hf_total_cost
                    }
                else:
                    comparison['huggingface'] = {
                        'request_count': 0,
                        'success_rate': 0,
                        'avg_response_time': 0,
                        'total_cost': 0
                    }
                
                return comparison
                
        except Exception as e:
            logger.error("Error obteniendo comparación de proveedores: %s", e)
            return {'error': str(e)}


# Instancia global del colector de métricas
_global_collector: Optional[HybridMetricsCollector] = None


def get_metrics_collector() -> HybridMetricsCollector:
    """Obtiene la instancia global del colector de métricas"""
    global _global_collector
    
    if _global_collector is None:
        _global_collector = HybridMetricsCollector()
    
    return _global_collector


def record_inference_metric(
    backend: str,
    agent_type: str,
    response_time_ms: float,
    success: bool,
    **kwargs
) -> None:
    """
    Función helper para registrar métricas de inferencia
    
    Args:
        backend: 'mlx' o 'huggingface'
        agent_type: tipo de agente ('sentiment', 'technical', etc.)
        response_time_ms: tiempo de respuesta en milisegundos
        success: si la inferencia fue exitosa
        **kwargs: parámetros adicionales (tokens, cost, etc.)
    """
    try:
        backend_enum = ModelBackend(backend.lower())
        agent_enum = AgentType(agent_type.lower())
        
        collector = get_metrics_collector()
        collector.record_metric(
            backend=backend_enum,
            agent_type=agent_enum,
            response_time_ms=response_time_ms,
            success=success,
            **kwargs
        )
        
    except (ValueError, Exception) as e:
        logger.error("Error registrando métrica: %s", e)


def get_agent_performance_report(agent_type: str) -> Dict[str, Any]:
    """Obtiene reporte de performance de un agente específico"""
    try:
        agent_enum = AgentType(agent_type.lower())
        collector = get_metrics_collector()
        metrics = collector.get_agent_metrics(agent_enum)
        
        return {
            'agent_type': agent_type,
            'total_requests': metrics.total_requests,
            'success_rate': metrics.success_rate,
            'performance_score': metrics.performance_score,
            'avg_response_time_ms': metrics.avg_response_time_ms,
            'total_cost_usd': metrics.total_cost_usd,
            'cache_hit_rate': metrics.cache_hit_rate,
            'mlx_stats': {
                'requests': metrics.mlx_requests,
                'success_rate': metrics.mlx_success_rate,
                'avg_time_ms': metrics.mlx_avg_time_ms
            },
            'hf_stats': {
                'requests': metrics.hf_requests,
                'success_rate': metrics.hf_success_rate,
                'avg_time_ms': metrics.hf_avg_time_ms
            }
        }
    except Exception as e:
        logger.error("Error obteniendo reporte de agente: %s", e)
        return {'error': str(e)}


def get_system_performance_report() -> Dict[str, Any]:
    """Obtiene reporte de performance del sistema completo"""
    try:
        collector = get_metrics_collector()
        return collector.get_comparison_report()
    except Exception as e:
        logger.error("Error obteniendo reporte del sistema: %s", e)
        return {'error': str(e)}


# Ejemplo de uso y testing
if __name__ == "__main__":
    import random
    
    print("🧪 Testing HybridMetricsCollector...")
    
    # Crear colector
    collector = HybridMetricsCollector(enable_dashboard=False)
    
    # Simular métricas
    agents = list(AgentType)
    backends = [ModelBackend.MLX, ModelBackend.HUGGINGFACE]
    
    for i in range(100):
        agent = random.choice(agents)
        backend = random.choice(backends)
        
        # Simular diferentes performance por backend
        if backend == ModelBackend.MLX:
            response_time = random.uniform(500, 2000)  # MLX más rápido
            success = random.random() > 0.05  # 95% éxito
            cost = 0.0  # MLX gratis
        else:
            response_time = random.uniform(1000, 4000)  # HF más lento
            success = random.random() > 0.1  # 90% éxito
            cost = random.uniform(0.001, 0.01)  # HF con costo
        
        collector.record_metric(
            backend=backend,
            agent_type=agent,
            response_time_ms=response_time,
            success=success,
            tokens_input=random.randint(50, 500),
            tokens_output=random.randint(20, 200),
            cost_usd=cost,
            cache_hit=random.random() > 0.7  # 30% cache hit
        )
    
    # Generar reporte
    report = collector.get_comparison_report()
    
    print("\n📊 REPORTE DE PRUEBA:")
    print(f"Total requests: {report['system_overview']['total_requests']}")
    print(f"Success rate: {report['system_overview']['success_rate']:.1f}%")
    print(f"MLX usage: {report['system_overview']['mlx_usage']:.1f}%")
    print(f"HF usage: {report['system_overview']['hf_usage']:.1f}%")
    print(f"Total cost: ${report['system_overview']['total_cost']:.3f}")
    
    print("\n🔥 Comparación Backend:")
    mlx_data = report['backend_comparison']['mlx']
    hf_data = report['backend_comparison']['huggingface']
    
    print(f"MLX - Success: {mlx_data['success_rate']:.1f}%, Time: {mlx_data['avg_response_time']:.0f}ms, Cost: ${mlx_data['total_cost']:.3f}")
    print(f"HF  - Success: {hf_data['success_rate']:.1f}%, Time: {hf_data['avg_response_time']:.0f}ms, Cost: ${hf_data['total_cost']:.3f}")
    
    print("\n🎯 Recomendaciones:")
    for rec in report['recommendations']:
        print(f"  {rec}")
    
    print("\n✅ HybridMetricsCollector test completado!")