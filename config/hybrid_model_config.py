"""
Configuración híbrida dinámica desde YAML.
"""
from dataclasses import dataclass
from typing import Literal, List, Optional, Dict, Any
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """
    Configuración de modelo
    
    Attributes:
        model_id: ID del modelo (MLX path o HF model ID)
        provider: Backend a usar ('mlx' o 'huggingface')
        temperature: Temperature para sampling
        max_tokens: Máximo tokens a generar
        priority: Prioridad (1=highest, se intenta primero)
        description: Descripción del modelo
    """
    model_id: str
    provider: Literal['mlx', 'huggingface']
    temperature: float = 0.1
    max_tokens: int = 1024
    priority: int = 1
    description: str = ""


def _map_provider(p: str) -> Optional[str]:
    q = (p or "").lower()
    if q in {"huggingface", "huggingface_inference"}:
        return "huggingface"
    if q in {"huggingface_mlx", "mlx"}:
        return "mlx"
    return None

def _load_yaml() -> Dict[str, Any]:
    path = Path(__file__).parent.parent / "config" / "llm_providers.yaml"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _build_configs_from_yaml() -> Dict[str, List[ModelConfig]]:
    data = _load_yaml()
    prof = data.get("active_profile")
    if not prof:
        return {}
    profile = data.get(prof) or {}
    out: Dict[str, List[ModelConfig]] = {}
    for agent, cfg in profile.items():
        if not isinstance(cfg, dict):
            continue
        prov = _map_provider(str(cfg.get("provider_type", "")))
        if prov is None:
            continue
        mc = ModelConfig(
            model_id=str(cfg.get("model_name", "")),
            provider=prov,
            temperature=float(cfg.get("temperature", 0.1)),
            max_tokens=int(cfg.get("max_tokens", 1024)),
            priority=1,
            description=""
        )
        out[agent] = [mc]
        fb_type = cfg.get("fallback_provider_type")
        fb_name = cfg.get("fallback_model_name")
        fb_prov = _map_provider(str(fb_type)) if fb_type else None
        if fb_prov and fb_name:
            out[agent].append(ModelConfig(
                model_id=str(fb_name),
                provider=fb_prov,
                temperature=float(cfg.get("temperature", 0.1)),
                max_tokens=int(cfg.get("max_tokens", 1024)),
                priority=2,
                description=""
            ))
    return out

AGENT_CONFIGS = _build_configs_from_yaml() or {}


# ============================================================================
# FUNCIONES DE ACCESO
# ============================================================================

def get_configs_for_agent(agent_type: str) -> List[ModelConfig]:
    """
    Obtener configuraciones para un agente (ordenadas por prioridad)
    
    Args:
        agent_type: Tipo de agente ('sentiment', 'technical', etc.)
        
    Returns:
        Lista de ModelConfig ordenada por prioridad (1=más alta)
    """
    configs = AGENT_CONFIGS.get(agent_type, [])
    
    if not configs:
        logger.warning(f"⚠️ No hay configuración para agente '{agent_type}'")
        return []
    
    # Ordenar por prioridad (1 es más alta)
    sorted_configs = sorted(configs, key=lambda c: c.priority)
    
    return sorted_configs


def get_primary_config(agent_type: str) -> Optional[ModelConfig]:
    """
    Obtener configuración primaria (priority=1) para un agente
    
    Args:
        agent_type: Tipo de agente
        
    Returns:
        ModelConfig primaria o None si no existe
    """
    configs = get_configs_for_agent(agent_type)
    
    if not configs:
        return None
    
    # Primera configuración es la de mayor prioridad
    return configs[0]


def get_provider_for_agent(agent_type: str) -> Optional[str]:
    """
    Obtener provider primario para un agente
    
    Args:
        agent_type: Tipo de agente
        
    Returns:
        'mlx', 'huggingface', o None
    """
    config = get_primary_config(agent_type)
    return config.provider if config else None


def should_use_huggingface(agent_type: str) -> bool:
    """
    Verificar si un agente debería usar HuggingFace como primario
    
    Args:
        agent_type: Tipo de agente
        
    Returns:
        True si HF es el provider primario
    """
    provider = get_provider_for_agent(agent_type)
    return provider == 'huggingface'


def get_fallback_configs(agent_type: str) -> List[ModelConfig]:
    """
    Obtener configuraciones de fallback (priority > 1)
    
    Args:
        agent_type: Tipo de agente
        
    Returns:
        Lista de configuraciones de fallback
    """
    configs = get_configs_for_agent(agent_type)
    
    # Filtrar solo fallbacks (priority > 1)
    fallbacks = [c for c in configs if c.priority > 1]
    
    return fallbacks


# ============================================================================
# RESUMEN DE CONFIGURACIÓN
# ============================================================================

def print_config_summary():
    """Imprimir resumen de la configuración híbrida"""
    print("\n" + "="*70)
    print("FENIX AI TRADING BOT - CONFIGURACIÓN HÍBRIDA MLX + HUGGINGFACE")
    print("="*70)
    
    for agent_type, configs in AGENT_CONFIGS.items():
        print(f"\n📊 {agent_type.upper()}")
        print("-" * 70)
        
        for i, config in enumerate(configs, 1):
            provider_emoji = "🌐" if config.provider == 'huggingface' else "🍎"
            print(f"  {i}. {provider_emoji} {config.provider.upper()}")
            print(f"     Model: {config.model_id}")
            print(f"     Temp: {config.temperature} | Max Tokens: {config.max_tokens}")
            print(f"     Priority: {config.priority} | {config.description}")
    
    print("\n" + "="*70)
    print("LEYENDA:")
    print("  🌐 = HuggingFace API (Cloud)")
    print("  🍎 = MLX Local (Mac M4)")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Mostrar configuración al ejecutar directamente
    print_config_summary()
