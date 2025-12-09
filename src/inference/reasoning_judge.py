from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.config.judge_config import JudgeModelConfig, get_judge_model_config
from src.inference.model_id_normalizer import normalize_model_id_for_provider
from src.inference.provider_registry import registry, setup_default_providers
from src.inference.providers.base import GenerationParams, ProviderError

logger = logging.getLogger(__name__)


@dataclass
class ReasoningJudgePayload:
    """Context passed to the LLM-as-a-judge."""

    agent_name: str
    prompt: str
    normalized_result: Dict[str, Any]
    raw_response: str
    backend: str
    metadata: Dict[str, Any]
    latency_ms: Optional[float] = None


@dataclass
class JudgeVerdict:
    """Structured verdict returned by the judge."""

    verdict: str
    score: float
    confidence: float
    critique: str
    success_estimate: Optional[bool] = None
    risks: List[str] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    raw_response: str = ""

    def as_entry_payload(self) -> Dict[str, Any]:
        """Flatten verdict so ReasoningBank can persist it."""

        return {
            "verdict": self.verdict,
            "score": self.score,
            "confidence": self.confidence,
            "notes": self.critique,
            "success_estimate": self.success_estimate,
            "tags": self.tags,
            "metadata": {
                "risks": self.risks,
                "improvements": self.improvements,
                "raw_response": self.raw_response,
            },
        }


class ReasoningLLMJudge:
    """LLM-powered judge that scores reasoning traces via existing providers."""

    def __init__(self, config: Optional[JudgeModelConfig] = None):
        self.config = config or get_judge_model_config()
        setup_default_providers()

    def evaluate(self, payload: ReasoningJudgePayload) -> Optional[JudgeVerdict]:
        provider = registry.get(self.config.provider)
        if not provider:
            logger.debug(
                "ReasoningLLMJudge: provider '%s' not registered, skipping",
                self.config.provider,
            )
            return None

        normalized_model_id = normalize_model_id_for_provider(
            self.config.model_id, self.config.provider
        )

        prompt = self._build_prompt(payload)
        params = GenerationParams(
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            extra={"system": self.config.system_prompt},
        )

        try:
            response = provider.generate_text(normalized_model_id or "", prompt, params)
        except ProviderError as exc:
            logger.debug("ReasoningLLMJudge provider error: %s", exc)
            # If rate-limited, attempt to find a fallback provider
            if exc.is_rate_limit:
                for name, p in registry.available().items():
                    if name == self.config.provider:
                        continue
                    try:
                        logger.debug("Attempting fallback judge provider: %s", name)
                        alt_model_id = normalize_model_id_for_provider(self.config.model_id, name)
                        response = p.generate_text(alt_model_id or "", prompt, params)
                        break
                    except ProviderError as e2:
                        logger.debug("Fallback provider %s error: %s", name, e2)
                        continue
                else:
                    return None
            else:
                return None
        except Exception as exc:  # pragma: no cover - safety net
            logger.debug("ReasoningLLMJudge unexpected error: %s", exc)
            return None

        text = (response.get("text") or "").strip()
        parsed = self._parse_response(text)
        if not parsed:
            return None

        success_estimate = parsed.get("success_estimate")
        if isinstance(success_estimate, str):
            lowered = success_estimate.strip().lower()
            if lowered in ("true", "yes", "approve", "approved"):
                success_estimate = True
            elif lowered in ("false", "no", "reject", "rejected"):
                success_estimate = False
            else:
                success_estimate = None

        verdict = JudgeVerdict(
            verdict=parsed.get("verdict", "inconclusive").lower(),
            score=float(parsed.get("score", 0.5)),
            confidence=float(parsed.get("confidence", 0.5)),
            critique=parsed.get("critique", "").strip(),
            success_estimate=success_estimate,
            risks=list(parsed.get("risks") or []),
            improvements=list(parsed.get("improvements") or []),
            tags=list(parsed.get("tags") or []),
            raw_response=text,
        )

        return verdict

    def _build_prompt(self, payload: ReasoningJudgePayload) -> str:
        norm_json = json.dumps(payload.normalized_result, ensure_ascii=False, indent=2)
        metadata_json = json.dumps(payload.metadata or {}, ensure_ascii=False, indent=2)
        raw = payload.raw_response
        if not isinstance(raw, str):
            raw = json.dumps(raw, ensure_ascii=False, indent=2)

        instruction = (
            "You are ReasoningBank-Judge. Evaluate the trading decision below and "
            "decide if the reasoning should be APPROVED, REJECTED or marked "
            "INCONCLUSIVE. Focus on internal coherence, signal alignment and risk "
            "controls. Output STRICT JSON with the schema described."
        )

        schema = (
            "{\n"
            '  "verdict": "approve|reject|inconclusive",\n'
            '  "success_estimate": true|false|null,\n'
            '  "score": 0.0-1.0,\n'
            '  "confidence": 0.0-1.0,\n'
            '  "critique": "short analysis",\n'
            '  "risks": ["list of concrete risks"],\n'
            '  "improvements": ["list of actionable improvements"],\n'
            '  "tags": ["optional labels"]\n'
            "}"
        )

        context = (
            f"### Agent Context\n"
            f"- Agent: {payload.agent_name}\n"
            f"- Backend: {payload.backend}\n"
            f"- LatencyMs: {payload.latency_ms}\n\n"
            f"### Task Prompt\n{payload.prompt}\n\n"
            f"### Normalized Result\n{norm_json}\n\n"
            f"### Raw Response\n{raw}\n\n"
            f"### Additional Metadata\n{metadata_json}\n\n"
            f"Return ONLY valid JSON matching this schema:\n{schema}\n"
        )

        return f"{instruction}\n\n{context}"

    def _parse_response(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON substring
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                logger.debug("ReasoningLLMJudge: no JSON detected in response.")
                return None
            snippet = text[start : end + 1]
            try:
                return json.loads(snippet)
            except Exception:
                logger.debug("ReasoningLLMJudge: failed to parse JSON snippet.")
                return None


_judge_instance: Optional[ReasoningLLMJudge] = None
_judge_lock = threading.Lock()


def get_reasoning_judge() -> Optional[ReasoningLLMJudge]:
    """Return a singleton judge instance, or None if initialization fails."""

    global _judge_instance
    if _judge_instance is None:
        with _judge_lock:
            if _judge_instance is None:
                try:
                    _judge_instance = ReasoningLLMJudge()
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("Unable to initialize ReasoningLLMJudge: %s", exc)
                    _judge_instance = None
    return _judge_instance
