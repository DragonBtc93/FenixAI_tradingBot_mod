"""
Base interfaces for unified inference providers.
Defines a common contract so agents can call a single API
regardless of the underlying backend (HF Inference, HF local,
MLX local, Ollama, OpenAI, etc.).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class GenerationParams:
    """Common generation parameters across providers."""
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stream: bool = False
    timeout: Optional[float] = 30.0
    priority: int = 1
    extra: Dict[str, Any] = field(default_factory=dict)


class ProviderError(Exception):
    """Raised when a provider fails to generate a response."""
    def __init__(self, message: str, *, is_rate_limit: bool = False, cause: Optional[Exception] = None):
        super().__init__(message)
        self.is_rate_limit = is_rate_limit
        self.cause = cause


class InferenceProvider(Protocol):
    """Protocol for inference providers. Implementations should be lightweight and stateless."""

    def name(self) -> str:
        """Provider identifier (e.g., 'hf_inference', 'mlx', 'ollama')."""
        ...

    def capabilities(self) -> Dict[str, bool]:
        """Return capability flags: supports_chat, supports_text, supports_vision, supports_tools."""
        ...

    def generate_text(self, model_id: str, prompt: str, params: GenerationParams) -> Dict[str, Any]:
        """
        Generate plain text from a `prompt`.
        Returns a dict with at least: {'text': str, 'metadata': {...}}.
        """
        ...

    def chat_completions(self, model_id: str, messages: List[Dict[str, str]], params: GenerationParams) -> Dict[str, Any]:
        """
        Chat-style completions using OpenAI-compatible messages format.
        Returns a dict with at least: {'text': str, 'metadata': {...}}.
        """
        ...

    def generate_with_vision(self, model_id: str, prompt: str, images: List[str], params: GenerationParams) -> Dict[str, Any]:
        """
        Multimodal generation with images.
        Returns a dict with at least: {'text': str, 'metadata': {...}}.
        """
        ...


def _metadata(backend: str, model_id: str, start_ts: float, tokens: Optional[int] = None, **extra: Any) -> Dict[str, Any]:
    """Helper to build metadata payload consistently."""
    latency_ms = (time.time() - start_ts) * 1000.0
    md = {
        'backend': backend,
        'model_id': model_id,
        'latency_ms': latency_ms,
    }
    if tokens is not None:
        md['tokens'] = tokens
    md.update(extra or {})
    return md