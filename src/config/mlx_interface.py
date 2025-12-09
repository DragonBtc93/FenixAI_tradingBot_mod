# config/mlx_interface.py
"""
MLX Interface - Provides compatibility layer for MLX models
"""

import logging
import asyncio
import os
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Check if MLX is disabled via environment variable
MLX_DISABLED = os.getenv('DISABLE_MLX', '0') == '1'

# MLX imports with fallback
if MLX_DISABLED:
    MLX_AVAILABLE = False
    mx = None
    load = None
    generate = None
    logging.getLogger(__name__).info("MLX deshabilitado por variable de entorno DISABLE_MLX")
else:
    try:
        import mlx.core as mx
        from mlx_lm.utils import load
        from mlx_lm.generate import generate
        MLX_AVAILABLE = True
        logging.getLogger(__name__).info("✅ MLX detectado y disponible correctamente")
    except ImportError as e:
        MLX_AVAILABLE = False
        mx = None
        load = None
        generate = None
        logging.getLogger(__name__).error(f"❌ MLX no disponible: {e}")

from .mlx_models import MODEL_CONFIGS

logger = logging.getLogger(__name__)

@dataclass
class MLXResponse:
    """Response from MLX model inference"""
    content: str
    model: str
    created_at: str
    done: bool

class MLXClient:
    """
    MLX Client that provides Ollama-compatible interface
    """
    
    def __init__(self):
        self.model_cache: Dict[str, tuple] = {}
        self.current_model: Optional[str] = None  # Track currently loaded model
        self.max_models_in_memory: int = 1  # Only allow 1 model at a time
        if not MLX_AVAILABLE:
            logger.warning("MLX not available. Running in compatibility mode.")
        else:
            logger.info("MLX Client initialized successfully")

        # Ensure MLX Home is configured to point to a folder with enough space.
        # Prefer the explicit env var MLX_HOME, fallback to XDG_CACHE_HOME or the repo cache.
        try:
            if not os.getenv('MLX_HOME'):
                suggested = os.path.join(os.path.abspath(os.getcwd()), 'cache', 'mlx')
                os.environ['MLX_HOME'] = suggested
                logger.info(f"⚠️ MLX_HOME unset, defaulting to: {suggested}")
        except Exception as e:
            logger.debug(f"Could not set MLX_HOME: {e}")

    def _unload_model(self, model_name: str):
        """Unload a specific model from memory"""
        if model_name in self.model_cache:
            logger.info(f"Unloading MLX model: {model_name}")
            del self.model_cache[model_name]
            if self.current_model == model_name:
                self.current_model = None

    def _clear_memory(self):
        """Clear all models from memory"""
        if self.model_cache:
            logger.info(f"Clearing all MLX models from memory: {list(self.model_cache.keys())}")
            self.model_cache.clear()
            self.current_model = None

    def _load_model(self, model_name: str) -> tuple:
        """Load MLX model and tokenizer with memory management"""
        # If this model is already loaded, return it
        if model_name in self.model_cache:
            logger.info(f"Using cached MLX model: {model_name}")
            self.current_model = model_name  # Update current model tracker
            return self.model_cache[model_name]
        
        if not MLX_AVAILABLE:
            raise RuntimeError("MLX not available")
        
        # If we have a different model loaded, clear it first (ONE MODEL AT A TIME)
        if self.model_cache and self.current_model != model_name:
            logger.info(f"🔄 Switching from {self.current_model} to {model_name} - clearing memory first")
            self._clear_memory()
        
        try:
            model_path = MODEL_CONFIGS[model_name]["path"]
            logger.info(f"🔧 Loading MLX model from path: {model_path}")
            
            if not MLX_AVAILABLE or load is None:
                raise RuntimeError("MLX load function not available")
                
            model, tokenizer = load(model_path)
            self.model_cache[model_name] = (model, tokenizer)
            self.current_model = model_name
            logger.info(f"✅ Successfully loaded model: {model_name}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"❌ Failed to load model {model_name}: {e}")
            raise

    def _generate_response(self, model_name: str, prompt: str, **kwargs) -> str:
        """Generate response using MLX model"""
        if not MLX_AVAILABLE:
            logger.error("MLX not available - cannot generate real response")
            return '{"status": "error", "reasoning": "MLX no está disponible"}'
        
        try:
            # Load the model (this will automatically unload others if needed)
            model, tokenizer = self._load_model(model_name)
            config = MODEL_CONFIGS.get(model_name, {})
            
            # Extract generation parameters from kwargs and config
            max_tokens = kwargs.get("max_tokens", config.get("max_tokens", 1024))
            
            logger.info(f"🤖 MLX: Generating with {model_name} (max_tokens: {max_tokens})")
            
            if not MLX_AVAILABLE or generate is None:
                raise RuntimeError("MLX generate function not available")
            
            # Set up generation arguments for **kwargs only - minimal set for compatibility
            gen_kwargs = {
                "max_tokens": max_tokens,
            }

            # Call generate with correct positional arguments
            response = generate(model, tokenizer, prompt, **gen_kwargs)
            
            logger.info(f"✅ MLX: Generated {len(response)} chars with {model_name}")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ MLX: Error generating response with {model_name}: {e}")
            return f'{{"status": "error", "message": "MLX Error: {str(e)[:100]}"}}'

    def chat(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> MLXResponse:
        """Chat completion compatible with Ollama interface"""
        try:
            prompt = self._convert_messages_to_prompt(messages)
            response_text = self._generate_response(model, prompt, **kwargs)
            return MLXResponse(
                content=response_text,
                model=model,
                created_at=str(time.time()),
                done=True,
            )
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            return MLXResponse(content=f"Error: {str(e)}", model=model, created_at=str(time.time()), done=True)

    def _convert_messages_to_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Convert OpenAI-style messages to prompt format"""
        prompt_parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        prompt_parts.append("<|im_start|>assistant\n")
        return "\n".join(prompt_parts)

    def generate(self, model: str, prompt: str, **kwargs) -> MLXResponse:
        """Generate completion compatible with Ollama interface"""
        try:
            response_text = self._generate_response(model, prompt, **kwargs)
            return MLXResponse(content=response_text, model=model, created_at=str(time.time()), done=True)
        except Exception as e:
            logger.error(f"Error in generate: {e}")
            return MLXResponse(content=f"Error: {str(e)}", model=model, created_at=str(time.time()), done=True)

    def force_memory_cleanup(self):
        """Force cleanup of all models to free memory"""
        if self.model_cache:
            logger.info("🧹 Forcing memory cleanup - unloading all MLX models")
            self._clear_memory()
            # Force garbage collection
            import gc
            gc.collect()
            # Additional MLX-specific cleanup
            if MLX_AVAILABLE and mx is not None:
                try:
                    mx.metal.clear_cache()
                    logger.info("🧹 MLX Metal cache cleared")
                except Exception as e:
                    logger.debug(f"Could not clear MLX metal cache: {e}")

    def prepare_for_vision_model(self, vision_model_name: str):
        """
        Prepara el sistema específicamente para cargar un modelo de visión.
        Descarga todos los otros modelos y limpia la memoria agresivamente.
        """
        logger.info(f"🎯 Preparando sistema para modelo de visión: {vision_model_name}")
        
        # 1. Forzar limpieza completa de memoria
        self.force_memory_cleanup()
        
        # 2. Esperar un momento para que la limpieza se complete
        import time
        time.sleep(1)
        
        # 3. Verificar que no hay modelos cargados
        if self.model_cache:
            logger.warning(f"⚠️ Modelos aún en memoria después de limpieza: {list(self.model_cache.keys())}")
            self._clear_memory()
        
        # 4. Verificar memoria del sistema
        memory_info = self.get_memory_info()
        logger.info(f"📊 Memoria del proceso: {memory_info['process_memory_mb']} MB")
        
        # 5. Si la memoria es muy alta, forzar garbage collection adicional
        if memory_info['process_memory_mb'] > 2000:  # Si usa más de 2GB
            logger.warning(f"⚠️ Memoria alta ({memory_info['process_memory_mb']} MB), forzando limpieza adicional")
            import gc
            for _ in range(3):
                gc.collect()
            time.sleep(0.5)
        
        logger.info(f"✅ Sistema preparado para modelo de visión: {vision_model_name}")

    def get_memory_status(self) -> dict:
        """Get current memory status"""
        return {
            "loaded_models": list(self.model_cache.keys()),
            "current_model": self.current_model,
            "models_count": len(self.model_cache),
            "max_models": self.max_models_in_memory
        }

    def get_memory_info(self) -> dict:
        """Get detailed memory information"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            "process_memory_mb": round(memory_info.rss / 1024 / 1024, 2),
            "loaded_models": list(self.model_cache.keys()),
            "current_model": self.current_model,
            "models_count": len(self.model_cache),
            "max_models": self.max_models_in_memory
        }

# Global client instance
_global_mlx_client = None

def get_client() -> MLXClient:
    """Get global MLX client instance (singleton pattern)"""
    global _global_mlx_client
    if _global_mlx_client is None:
        _global_mlx_client = MLXClient()
        logger.info("🔧 Created global MLX client instance")
    return _global_mlx_client

def reset_global_client():
    """Reset global client instance (useful for testing)"""
    global _global_mlx_client
    if _global_mlx_client:
        _global_mlx_client.force_memory_cleanup()
    _global_mlx_client = None
    logger.info("🔄 Reset global MLX client instance")