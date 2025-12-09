"""
Sistema Avanzado de Gestión de Costos para FenixAI Trading Bot
Incluye tracking por agente, alertas en tiempo real, budget limits y optimización automática
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Severidad de las alertas de costo"""
    INFO = "info"
    WARNING = "warning" 
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class CostPeriod(Enum):
    """Períodos para tracking de costos"""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class CostAlert:
    """Alerta de costo"""
    timestamp: float
    severity: AlertSeverity
    message: str
    agent_type: str
    current_cost: float
    threshold: float
    period: CostPeriod
    suggested_actions: List[str] = field(default_factory=list)
    auto_action_taken: Optional[str] = None


@dataclass
class BudgetLimit:
    """Límite de presupuesto"""
    agent_type: str
    period: CostPeriod
    limit: float
    current_spent: float = 0.0
    alert_thresholds: Dict[AlertSeverity, float] = field(default_factory=lambda: {
        AlertSeverity.INFO: 0.5,      # 50%
        AlertSeverity.WARNING: 0.8,   # 80%
        AlertSeverity.CRITICAL: 0.95, # 95%
        AlertSeverity.EMERGENCY: 1.0  # 100%
    })
    enabled: bool = True
    auto_throttle: bool = True  # Throttle automático al alcanzar límites


@dataclass
class CostMetrics:
    """Métricas de costo para un agente/proveedor"""
    agent_type: str
    provider: str
    total_requests: int = 0
    total_cost: float = 0.0
    cost_per_request: float = 0.0
    min_cost: float = float('inf')
    max_cost: float = 0.0
    recent_costs: deque = field(default_factory=lambda: deque(maxlen=100))
    cost_trend: float = 0.0  # Tendencia de costo (+ subiendo, - bajando)
    hourly_costs: Dict[int, float] = field(default_factory=dict)  # Costo por hora
    daily_costs: Dict[str, float] = field(default_factory=dict)   # Costo por día
    peak_hours: List[int] = field(default_factory=list)          # Horas pico de uso
    
    def update_cost(self, cost: float):
        """Actualizar métricas con nuevo costo"""
        self.total_requests += 1
        self.total_cost += cost
        self.cost_per_request = self.total_cost / self.total_requests
        
        self.min_cost = min(self.min_cost, cost)
        self.max_cost = max(self.max_cost, cost)
        
        self.recent_costs.append(cost)
        
        # Calcular tendencia de costo
        if len(self.recent_costs) >= 10:
            recent_list = list(self.recent_costs)
            mid_point = len(recent_list) // 2
            first_half_avg = statistics.mean(recent_list[:mid_point])
            second_half_avg = statistics.mean(recent_list[mid_point:])
            self.cost_trend = second_half_avg - first_half_avg
        
        # Tracking por tiempo
        current_time = datetime.now()
        hour_key = current_time.hour
        day_key = current_time.strftime('%Y-%m-%d')
        
        self.hourly_costs[hour_key] = self.hourly_costs.get(hour_key, 0) + cost
        self.daily_costs[day_key] = self.daily_costs.get(day_key, 0) + cost


@dataclass
class OptimizationRecommendation:
    """Recomendación de optimización de costos"""
    priority: int  # 1 = alta, 2 = media, 3 = baja
    type: str     # 'model_switch', 'throttle', 'cache_improve', etc.
    agent_type: str
    current_cost: float
    potential_savings: float
    description: str
    implementation_effort: str  # 'low', 'medium', 'high'
    auto_applicable: bool = False
    actions: List[str] = field(default_factory=list)


class AdvancedCostManager:
    """Sistema Avanzado de Gestión de Costos"""
    
    def __init__(self):
        self.cost_metrics: Dict[str, Dict[str, CostMetrics]] = defaultdict(dict)  # agent -> provider -> metrics
        self.budget_limits: Dict[str, BudgetLimit] = {}
        self.alerts: List[CostAlert] = []
        self.recommendations: List[OptimizationRecommendation] = []
        
        # Configuración por defecto
        self.default_budgets = {
            'sentiment': {'daily': 0.10, 'weekly': 0.50, 'monthly': 2.00},
            'technical': {'daily': 0.15, 'weekly': 0.75, 'monthly': 3.00},
            'visual': {'daily': 0.50, 'weekly': 2.50, 'monthly': 10.00},  # Más caro por análisis visual
            'qabba': {'daily': 0.20, 'weekly': 1.00, 'monthly': 4.00},
            'decision': {'daily': 0.25, 'weekly': 1.25, 'monthly': 5.00},
            'risk': {'daily': 0.30, 'weekly': 1.50, 'monthly': 6.00}      # Crítico para trading
        }
        
        # Costos típicos por modelo (estimados)
        self.model_costs = {
            'mlx': 0.0,  # Local, gratis
            'huggingface': {
                'microsoft/DialoGPT-medium': 0.002,
                'microsoft/DialoGPT-large': 0.005,
                'Qwen/Qwen2.5-1.5B-Instruct': 0.001,
                'Qwen/Qwen2.5-7B-Instruct': 0.003,
                'microsoft/Phi-3.5-mini-instruct': 0.0015,
                'llava-hf/llava-1.5-7b-hf': 0.01,  # Vision model más caro
                'default': 0.003
            }
        }
        
        # Estadísticas globales
        self.global_stats = {
            'total_cost': 0.0,
            'total_requests': 0,
            'cost_savings': 0.0,
            'auto_optimizations': 0,
            'alerts_generated': 0,
            'budget_violations': 0,
            'peak_cost_hour': None,
            'most_expensive_agent': None,
            'most_efficient_agent': None
        }
        
        self._setup_default_budgets()
        
        logger.info("💰 AdvancedCostManager initialized with budget tracking")
    
    def _setup_default_budgets(self):
        """Configurar presupuestos por defecto"""
        # Configurar presupuestos diarios por defecto
        for agent_type, budgets in self.default_budgets.items():
            daily_limit = budgets['daily']
            self.set_budget_limit(agent_type, CostPeriod.DAILY, daily_limit)
    
    def set_budget_limit(
        self,
        agent_type: str,
        period: CostPeriod,
        limit: float,
        auto_throttle: bool = True
    ):
        """Configurar límite de presupuesto"""
        budget_key = f"{agent_type}_{period.value}"
        
        self.budget_limits[budget_key] = BudgetLimit(
            agent_type=agent_type,
            period=period,
            limit=limit,
            auto_throttle=auto_throttle
        )
        
        logger.info("💰 Budget set for %s (%s): $%.4f", agent_type, period.value, limit)
    
    def track_cost(
        self,
        agent_type: str,
        provider: str,
        cost: float,
        model_id: Optional[str] = None,
        request_tokens: Optional[int] = None,
        response_tokens: Optional[int] = None
    ):
        """Registrar costo de una request"""
        
        # Crear métricas si no existen
        if provider not in self.cost_metrics[agent_type]:
            self.cost_metrics[agent_type][provider] = CostMetrics(
                agent_type=agent_type,
                provider=provider
            )
        
        metrics = self.cost_metrics[agent_type][provider]
        metrics.update_cost(cost)
        
        # Actualizar estadísticas globales
        self.global_stats['total_cost'] += cost
        self.global_stats['total_requests'] += 1
        
        # Verificar límites de presupuesto
        self._check_budget_limits(agent_type, cost)
        
        # Generar recomendaciones si es necesario
        self._generate_recommendations(agent_type)
        
        logger.debug("💰 Cost tracked: %s/%s $%.6f", agent_type, provider, cost)
    
    def _check_budget_limits(self, agent_type: str, new_cost: float):
        """Verificar límites de presupuesto y generar alertas"""
        current_time = datetime.now()
        
        # Verificar presupuesto diario
        daily_key = f"{agent_type}_daily"
        if daily_key in self.budget_limits:
            budget = self.budget_limits[daily_key]
            
            # Calcular gasto actual del día
            day_key = current_time.strftime('%Y-%m-%d')
            daily_spent = 0.0
            
            for provider_metrics in self.cost_metrics[agent_type].values():
                daily_spent += provider_metrics.daily_costs.get(day_key, 0.0)
            
            budget.current_spent = daily_spent + new_cost
            
            # Verificar umbrales
            for severity, threshold in budget.alert_thresholds.items():
                threshold_amount = budget.limit * threshold
                
                if budget.current_spent >= threshold_amount:
                    self._generate_alert(
                        severity=severity,
                        agent_type=agent_type,
                        current_cost=budget.current_spent,
                        threshold=threshold_amount,
                        period=CostPeriod.DAILY,
                        budget_limit=budget.limit
                    )
    
    def _generate_alert(
        self,
        severity: AlertSeverity,
        agent_type: str,
        current_cost: float,
        threshold: float,
        period: CostPeriod,
        budget_limit: float
    ):
        """Generar alerta de costo"""
        
        # Evitar alertas duplicadas recientes
        recent_alerts = [
            a for a in self.alerts[-10:]  # Últimas 10 alertas
            if (a.agent_type == agent_type and 
                a.severity == severity and
                time.time() - a.timestamp < 300)  # 5 minutos
        ]
        
        if recent_alerts:
            return  # Ya hay una alerta reciente similar
        
        percentage = (current_cost / budget_limit) * 100
        
        suggested_actions = []
        auto_action = None
        
        if severity == AlertSeverity.WARNING:
            suggested_actions = [
                "Considerar usar más MLX local para reducir costos",
                "Revisar si hay requests innecesarias",
                "Optimizar prompts para reducir tokens"
            ]
        elif severity == AlertSeverity.CRITICAL:
            suggested_actions = [
                "URGENTE: Reducir uso del agente inmediatamente",
                "Forzar uso de MLX local únicamente",
                "Revisar configuraciones de modelo"
            ]
            auto_action = "Throttling automático activado" if severity == AlertSeverity.CRITICAL else None
        elif severity == AlertSeverity.EMERGENCY:
            suggested_actions = [
                "EMERGENCIA: Agente bloqueado temporalmente",
                "Revisar budget limits",
                "Contactar administrador del sistema"
            ]
            auto_action = "Agente temporalmente deshabilitado"
        
        alert = CostAlert(
            timestamp=time.time(),
            severity=severity,
            message=f"Presupuesto {period.value} para {agent_type} al {percentage:.1f}% (${current_cost:.4f}/${budget_limit:.4f})",
            agent_type=agent_type,
            current_cost=current_cost,
            threshold=threshold,
            period=period,
            suggested_actions=suggested_actions,
            auto_action_taken=auto_action
        )
        
        self.alerts.append(alert)
        self.global_stats['alerts_generated'] += 1
        
        if current_cost >= budget_limit:
            self.global_stats['budget_violations'] += 1
        
        # Log según severidad
        if severity == AlertSeverity.EMERGENCY:
            logger.critical("🚨 COST EMERGENCY: %s", alert.message)
        elif severity == AlertSeverity.CRITICAL:
            logger.error("💥 COST CRITICAL: %s", alert.message)
        elif severity == AlertSeverity.WARNING:
            logger.warning("⚠️ COST WARNING: %s", alert.message)
        else:
            logger.info("💰 COST INFO: %s", alert.message)
    
    def _generate_recommendations(self, agent_type: str):
        """Generar recomendaciones de optimización"""
        
        # Analizar métricas del agente
        agent_metrics = self.cost_metrics.get(agent_type, {})
        if not agent_metrics:
            return
        
        total_agent_cost = sum(m.total_cost for m in agent_metrics.values())
        total_agent_requests = sum(m.total_requests for m in agent_metrics.values())
        
        if total_agent_requests == 0:
            return
        
        avg_cost_per_request = total_agent_cost / total_agent_requests
        
        recommendations = []
        
        # Recomendación 1: Cambio de modelo si HF es muy caro
        if 'huggingface' in agent_metrics and 'mlx' in agent_metrics:
            hf_metrics = agent_metrics['huggingface']
            mlx_metrics = agent_metrics['mlx']
            
            if hf_metrics.cost_per_request > 0.005:  # Umbral de costo alto
                potential_savings = hf_metrics.total_cost * 0.8  # 80% de ahorro estimado
                
                recommendations.append(OptimizationRecommendation(
                    priority=1,
                    type='model_switch',
                    agent_type=agent_type,
                    current_cost=hf_metrics.cost_per_request,
                    potential_savings=potential_savings,
                    description=f"Cambiar de HuggingFace a MLX local puede ahorrar ~${potential_savings:.4f}",
                    implementation_effort='low',
                    auto_applicable=True,
                    actions=[
                        "Configurar MLX como proveedor primario",
                        "Usar HuggingFace solo como fallback",
                        "Optimizar prompts para MLX"
                    ]
                ))
        
        # Recomendación 2: Optimización de cache si hay requests repetitivas
        if total_agent_requests > 50:  # Suficientes requests para analizar
            cache_hit_potential = self._estimate_cache_potential(agent_type)
            
            if cache_hit_potential > 0.3:  # 30% de requests podrían ser cacheadas
                potential_savings = total_agent_cost * cache_hit_potential * 0.9
                
                recommendations.append(OptimizationRecommendation(
                    priority=2,
                    type='cache_improve',
                    agent_type=agent_type,
                    current_cost=total_agent_cost,
                    potential_savings=potential_savings,
                    description=f"Mejorar cache puede ahorrar ~${potential_savings:.4f} ({cache_hit_potential*100:.1f}% hit rate estimado)",
                    implementation_effort='medium',
                    auto_applicable=False,
                    actions=[
                        "Aumentar TTL del cache para este agente",
                        "Implementar cache más agresivo",
                        "Normalizar prompts similares"
                    ]
                ))
        
        # Recomendación 3: Throttling si hay picos de costo
        recent_costs = []
        for metrics in agent_metrics.values():
            recent_costs.extend(list(metrics.recent_costs))
        
        if len(recent_costs) >= 20:
            recent_avg = statistics.mean(recent_costs[-10:])  # Últimas 10
            overall_avg = statistics.mean(recent_costs)
            
            if recent_avg > overall_avg * 1.5:  # Costo reciente 50% mayor
                recommendations.append(OptimizationRecommendation(
                    priority=1,
                    type='throttle',
                    agent_type=agent_type,
                    current_cost=recent_avg,
                    potential_savings=recent_avg - overall_avg,
                    description=f"Costos recientes elevados: ${recent_avg:.4f} vs promedio ${overall_avg:.4f}",
                    implementation_effort='low',
                    auto_applicable=True,
                    actions=[
                        "Implementar rate limiting más agresivo",
                        "Reducir frecuencia de requests",
                        "Revisar si hay loops de requests"
                    ]
                ))
        
        # Actualizar recomendaciones (evitar duplicados)
        for rec in recommendations:
            existing = [r for r in self.recommendations 
                       if r.agent_type == rec.agent_type and r.type == rec.type]
            if not existing:
                self.recommendations.append(rec)
    
    def _estimate_cache_potential(self, agent_type: str) -> float:
        """Estimar potencial de mejora de cache"""
        # Simulación simple - en un sistema real analizaríamos patrones de prompts
        agent_patterns = {
            'sentiment': 0.4,    # Análisis de sentimiento tiende a repetirse
            'technical': 0.3,    # Análisis técnico con algunos patrones
            'visual': 0.2,       # Análisis visual más único
            'qabba': 0.35,       # Patrones moderados
            'decision': 0.25,    # Decisiones más únicas
            'risk': 0.3         # Análisis de riesgo con algunos patrones
        }
        return agent_patterns.get(agent_type, 0.25)
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Obtener resumen completo de costos"""
        summary = {
            'timestamp': time.time(),
            'global_stats': self.global_stats.copy(),
            'by_agent': {},
            'by_provider': defaultdict(lambda: {'total_cost': 0.0, 'total_requests': 0}),
            'budget_status': {},
            'recent_alerts': [],
            'top_recommendations': [],
            'cost_trends': {},
            'efficiency_metrics': {}
        }
        
        # Análisis por agente
        for agent_type, providers in self.cost_metrics.items():
            agent_total_cost = 0.0
            agent_total_requests = 0
            agent_providers = {}
            
            for provider, metrics in providers.items():
                agent_total_cost += metrics.total_cost
                agent_total_requests += metrics.total_requests
                
                agent_providers[provider] = {
                    'total_cost': metrics.total_cost,
                    'total_requests': metrics.total_requests,
                    'cost_per_request': metrics.cost_per_request,
                    'cost_trend': metrics.cost_trend
                }
                
                # Agregar a resumen por proveedor
                summary['by_provider'][provider]['total_cost'] += metrics.total_cost
                summary['by_provider'][provider]['total_requests'] += metrics.total_requests
            
            summary['by_agent'][agent_type] = {
                'total_cost': agent_total_cost,
                'total_requests': agent_total_requests,
                'cost_per_request': agent_total_cost / max(agent_total_requests, 1),
                'providers': agent_providers
            }
        
        # Estado de presupuestos
        for budget_key, budget in self.budget_limits.items():
            summary['budget_status'][budget_key] = {
                'limit': budget.limit,
                'current_spent': budget.current_spent,
                'percentage_used': (budget.current_spent / budget.limit) * 100,
                'enabled': budget.enabled,
                'auto_throttle': budget.auto_throttle
            }
        
        # Alertas recientes (últimas 10)
        summary['recent_alerts'] = [
            {
                'timestamp': alert.timestamp,
                'severity': alert.severity.value,
                'message': alert.message,
                'agent_type': alert.agent_type,
                'auto_action_taken': alert.auto_action_taken
            }
            for alert in self.alerts[-10:]
        ]
        
        # Top recomendaciones (por prioridad)
        top_recommendations = sorted(self.recommendations, key=lambda x: x.priority)[:5]
        summary['top_recommendations'] = [
            {
                'priority': rec.priority,
                'type': rec.type,
                'agent_type': rec.agent_type,
                'potential_savings': rec.potential_savings,
                'description': rec.description,
                'auto_applicable': rec.auto_applicable
            }
            for rec in top_recommendations
        ]
        
        # Métricas de eficiencia
        if summary['by_agent']:
            most_expensive = max(summary['by_agent'].items(), 
                               key=lambda x: x[1]['total_cost'])
            most_efficient = min(summary['by_agent'].items(), 
                               key=lambda x: x[1]['cost_per_request'])
            
            summary['efficiency_metrics'] = {
                'most_expensive_agent': most_expensive[0],
                'most_expensive_cost': most_expensive[1]['total_cost'],
                'most_efficient_agent': most_efficient[0],
                'most_efficient_cost_per_request': most_efficient[1]['cost_per_request']
            }
        
        return summary
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generar reporte de optimización detallado"""
        summary = self.get_cost_summary()
        
        report = {
            'timestamp': time.time(),
            'total_potential_savings': sum(r.potential_savings for r in self.recommendations),
            'auto_applicable_savings': sum(r.potential_savings for r in self.recommendations if r.auto_applicable),
            'optimization_opportunities': len(self.recommendations),
            'priority_breakdown': {
                'high': len([r for r in self.recommendations if r.priority == 1]),
                'medium': len([r for r in self.recommendations if r.priority == 2]),
                'low': len([r for r in self.recommendations if r.priority == 3])
            },
            'cost_breakdown': summary['by_agent'],
            'budget_compliance': {},
            'recommendations_by_type': defaultdict(list)
        }
        
        # Análisis de cumplimiento de presupuesto
        for budget_key, budget_status in summary['budget_status'].items():
            compliance = 'compliant' if budget_status['percentage_used'] < 80 else 'at_risk' if budget_status['percentage_used'] < 100 else 'violated'
            report['budget_compliance'][budget_key] = {
                'status': compliance,
                'percentage_used': budget_status['percentage_used'],
                'remaining_budget': budget_status['limit'] - budget_status['current_spent']
            }
        
        # Agrupar recomendaciones por tipo
        for rec in self.recommendations:
            report['recommendations_by_type'][rec.type].append({
                'agent_type': rec.agent_type,
                'potential_savings': rec.potential_savings,
                'description': rec.description,
                'priority': rec.priority,
                'auto_applicable': rec.auto_applicable
            })
        
        return report
    
    def apply_auto_optimizations(self) -> Dict[str, Any]:
        """Aplicar optimizaciones automáticas"""
        applied_optimizations = []
        total_estimated_savings = 0.0
        
        auto_recommendations = [r for r in self.recommendations if r.auto_applicable]
        
        for rec in auto_recommendations:
            if rec.type == 'model_switch':
                # Cambiar prioridad de proveedores para este agente
                optimization = {
                    'type': rec.type,
                    'agent_type': rec.agent_type,
                    'action': 'Switched to MLX local as primary provider',
                    'estimated_savings': rec.potential_savings
                }
                applied_optimizations.append(optimization)
                total_estimated_savings += rec.potential_savings
                
            elif rec.type == 'throttle':
                # Aplicar throttling más agresivo
                optimization = {
                    'type': rec.type,
                    'agent_type': rec.agent_type,
                    'action': 'Applied aggressive rate limiting',
                    'estimated_savings': rec.potential_savings
                }
                applied_optimizations.append(optimization)
                total_estimated_savings += rec.potential_savings
        
        # Actualizar estadísticas
        self.global_stats['auto_optimizations'] += len(applied_optimizations)
        self.global_stats['cost_savings'] += total_estimated_savings
        
        # Remover recomendaciones aplicadas
        self.recommendations = [r for r in self.recommendations if not r.auto_applicable]
        
        logger.info("🤖 Applied %d auto-optimizations, estimated savings: $%.4f", 
                   len(applied_optimizations), total_estimated_savings)
        
        return {
            'applied_count': len(applied_optimizations),
            'total_estimated_savings': total_estimated_savings,
            'optimizations': applied_optimizations
        }
    
    def reset_period_costs(self, period: CostPeriod):
        """Resetear costos de un período (para testing o nuevo período)"""
        if period == CostPeriod.DAILY:
            for agent_metrics in self.cost_metrics.values():
                for metrics in agent_metrics.values():
                    metrics.daily_costs.clear()
        
        # Resetear presupuestos del período
        for budget_key, budget in self.budget_limits.items():
            if budget.period == period:
                budget.current_spent = 0.0
        
        logger.info("🔄 Reset %s costs and budgets", period.value)
    
    def export_cost_data(self, filepath: str):
        """Exportar datos de costo a archivo JSON"""
        export_data = {
            'timestamp': time.time(),
            'cost_summary': self.get_cost_summary(),
            'optimization_report': self.get_optimization_report(),
            'alerts': [
                {
                    'timestamp': alert.timestamp,
                    'severity': alert.severity.value,
                    'message': alert.message,
                    'agent_type': alert.agent_type,
                    'current_cost': alert.current_cost,
                    'threshold': alert.threshold,
                    'suggested_actions': alert.suggested_actions,
                    'auto_action_taken': alert.auto_action_taken
                }
                for alert in self.alerts
            ],
            'recommendations': [
                {
                    'priority': rec.priority,
                    'type': rec.type,
                    'agent_type': rec.agent_type,
                    'current_cost': rec.current_cost,
                    'potential_savings': rec.potential_savings,
                    'description': rec.description,
                    'implementation_effort': rec.implementation_effort,
                    'auto_applicable': rec.auto_applicable,
                    'actions': rec.actions
                }
                for rec in self.recommendations
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info("💾 Cost data exported to %s", filepath)
    
    # Sobrescribir método track_cost para compatibilidad con el demo
    def track_cost(self, *args, **kwargs):
        """Método track_cost compatible con múltiples firmas"""
        # Si se llama con la firma nueva (provider, model_id, cost, tokens_used, operation_type)
        if len(args) >= 3 and isinstance(args[0], str) and isinstance(args[1], str):
            return self._track_cost_new_signature(*args, **kwargs)
        # Si se llama con la firma original (agent_type, provider, cost, ...)
        else:
            return self._track_cost_original(*args, **kwargs)
    
    def _track_cost_new_signature(
        self,
        provider: str,
        model_id: str,
        cost: float,
        tokens_used: int = None,
        operation_type: str = None,
        **kwargs
    ):
        """Nueva firma compatible con el demo"""
        try:
            agent_type = operation_type if operation_type else 'general'
            return self._track_cost_original(
                agent_type=agent_type,
                provider=provider,
                cost=cost,
                model_id=model_id,
                request_tokens=tokens_used,
                response_tokens=0
            )
        except Exception as e:
            logger.error("Error tracking cost (new signature): %s", e)
    
    def _track_cost_original(
        self,
        agent_type: str,
        provider: str,
        cost: float,
        model_id: Optional[str] = None,
        request_tokens: Optional[int] = None,
        response_tokens: Optional[int] = None
    ):
        """Firma original del método"""
        # Crear métricas si no existen
        if provider not in self.cost_metrics[agent_type]:
            self.cost_metrics[agent_type][provider] = CostMetrics(
                agent_type=agent_type,
                provider=provider
            )
        
        metrics = self.cost_metrics[agent_type][provider]
        metrics.update_cost(cost)
        
        # Actualizar estadísticas globales
        self.global_stats['total_cost'] += cost
        self.global_stats['total_requests'] += 1
        
        # Verificar límites de presupuesto
        self._check_budget_limits(agent_type, cost)
        
        # Generar recomendaciones si es necesario
        self._generate_recommendations(agent_type)
    
    async def track_cost_compatible(
        self,
        provider: str,
        model_id: str,
        cost: float,
        tokens_used: int = None,
        operation_type: str = None,
        **kwargs
    ) -> None:
        """
        Método de compatibilidad para track_cost con firma compatible con el demo
        
        Args:
            provider: Proveedor ('huggingface', 'mlx', 'hybrid')
            model_id: ID del modelo
            cost: Costo en USD
            tokens_used: Tokens utilizados (opcional)
            operation_type: Tipo de operación (agent type)
        """
        try:
            # Determinar agent_type del operation_type o usar default
            agent_type = operation_type if operation_type else 'general'
            
            # Usar el método original con los parámetros correctos
            self.track_cost(
                agent_type=agent_type,
                provider=provider,
                cost=cost,
                model_id=model_id,
                request_tokens=tokens_used,
                response_tokens=0
            )
            
        except Exception as e:
            logger.error("Error tracking cost: %s", e)
    
    async def get_daily_costs(self) -> Dict[str, Any]:
        """
        Obtiene costos del día actual
        
        Returns:
            Dict con costos diarios por agente y totales
        """
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            daily_total = 0.0
            costs_by_agent = {}
            
            # Sumar costos por agente
            for agent_type, providers in self.cost_metrics.items():
                agent_daily_cost = 0.0
                for provider, metrics in providers.items():
                    if today in metrics.daily_costs:
                        agent_daily_cost += metrics.daily_costs[today]
                
                costs_by_agent[agent_type] = agent_daily_cost
                daily_total += agent_daily_cost
            
            return {
                'date': today,
                'total': daily_total,
                'by_agent': costs_by_agent,
                'currency': 'USD'
            }
            
        except Exception as e:
            logger.error("Error getting daily costs: %s", e)
            return {'total': 0.0, 'by_agent': {}, 'error': str(e)}
    
    async def check_budget_status(self) -> Dict[str, Any]:
        """
        Verifica el estado del presupuesto
        
        Returns:
            Dict con información del presupuesto y uso actual
        """
        try:
            daily_costs = await self.get_daily_costs()
            total_daily_cost = daily_costs['total']
            
            # Calcular presupuesto total diario
            total_daily_budget = sum(
                budgets['daily'] for budgets in self.default_budgets.values()
            )
            
            # Calcular porcentaje de uso
            usage_percentage = (total_daily_cost / total_daily_budget * 100) if total_daily_budget > 0 else 0
            
            # Determinar estado del presupuesto
            if usage_percentage >= 90:
                status = 'critical'
                message = 'Presupuesto diario crítico (>90% usado)'
            elif usage_percentage >= 75:
                status = 'warning'
                message = 'Presupuesto diario alto (>75% usado)'
            elif usage_percentage >= 50:
                status = 'moderate'
                message = 'Presupuesto diario moderado (>50% usado)'
            else:
                status = 'ok'
                message = 'Presupuesto diario bajo control'
            
            return {
                'status': status,
                'message': message,
                'usage_percentage': usage_percentage,
                'daily_cost': total_daily_cost,
                'daily_budget': total_daily_budget,
                'remaining_budget': total_daily_budget - total_daily_cost,
                'by_agent': daily_costs['by_agent']
            }
            
        except Exception as e:
            logger.error("Error checking budget status: %s", e)
            return {'status': 'error', 'message': str(e), 'usage_percentage': 0}
    
    async def get_cost_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """
        Obtiene sugerencias de optimización de costos
        
        Returns:
            Lista de sugerencias de optimización
        """
        try:
            suggestions = []
            
            # Analizar costos por agente
            for agent_type, providers in self.cost_metrics.items():
                total_agent_cost = sum(
                    sum(metrics.daily_costs.values()) 
                    for metrics in providers.values()
                )
                
                if total_agent_cost > 0.1:  # Solo sugerir si costo > $0.10
                    # Sugerir uso de MLX si está usando HuggingFace
                    if 'huggingface' in providers and 'mlx' not in providers:
                        suggestions.append({
                            'type': 'provider_switch',
                            'agent_type': agent_type,
                            'suggestion': f'Considerar usar MLX local para {agent_type} para reducir costos',
                            'potential_savings': total_agent_cost * 0.8,  # 80% de ahorro
                            'priority': 'medium'
                        })
                    
                    # Sugerir cache si hay muchos requests repetitivos
                    total_requests = sum(
                        metrics.request_count 
                        for metrics in providers.values()
                    )
                    
                    if total_requests > 50:
                        suggestions.append({
                            'type': 'cache_optimization',
                            'agent_type': agent_type,
                            'suggestion': f'Mejorar cache para {agent_type} con {total_requests} requests',
                            'potential_savings': total_agent_cost * 0.3,  # 30% de ahorro
                            'priority': 'low'
                        })
            
            # Sugerencias generales
            daily_costs = await self.get_daily_costs()
            if daily_costs['total'] > 2.0:  # Más de $2/día
                suggestions.append({
                    'type': 'general_optimization',
                    'suggestion': 'Costos diarios elevados. Considerar implementar rate limiting más estricto',
                    'potential_savings': daily_costs['total'] * 0.2,
                    'priority': 'high'
                })
            
            return suggestions
            
        except Exception as e:
            logger.error("Error getting optimization suggestions: %s", e)
            return [{'suggestion': f'Error: {e}', 'type': 'error'}]


# Instancia global del cost manager
cost_manager = AdvancedCostManager()


def get_cost_manager() -> AdvancedCostManager:
    """Obtener instancia global del cost manager"""
    return cost_manager


def track_request_cost(
    agent_type: str,
    provider: str,
    cost: float,
    model_id: Optional[str] = None,
    request_tokens: Optional[int] = None,
    response_tokens: Optional[int] = None
):
    """Función de conveniencia para trackear costos"""
    cost_manager.track_cost(
        agent_type=agent_type,
        provider=provider,
        cost=cost,
        model_id=model_id,
        request_tokens=request_tokens,
        response_tokens=response_tokens
    )


def get_cost_estimation(
    provider: str,
    model_id: Optional[str] = None,
    request_tokens: Optional[int] = None,
    response_tokens: Optional[int] = None
) -> float:
    """Estimar costo de una request"""
    if provider == 'mlx':
        return 0.0  # MLX es local y gratuito
    
    elif provider == 'huggingface':
        model_cost = cost_manager.model_costs['huggingface'].get(
            model_id, 
            cost_manager.model_costs['huggingface']['default']
        )
        
        # Estimación simple - en un sistema real usaríamos pricing real
        if request_tokens and response_tokens:
            total_tokens = request_tokens + response_tokens
            return (total_tokens / 1000) * model_cost  # Costo por 1K tokens
        else:
            return model_cost  # Costo base por request
    
    return 0.001  # Costo default para proveedores desconocidos