"""
Vision LLM client - wrap OpenAI's chat API with:
. JSON model
. Timeout per call
. Retry with backoff on transient errors
. Token/cost logging
. Response validation
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    RateLimitError,
)

from prompts import HELMET_CHECK_PROMPT

log = logging.getLogger(__name__)

# Transient error classes we retry on. Authentication and 400s are not here
# because they won't succeed on retry.
_RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError)


class OpenAIVisionClient:
    """
    analyze(image_b64) -> verdict dict.

    Returns:
        {
          "unsafe": bool,
          "summary": str,
          "persons": [{"track_id": int|None, "helmet": bool|None,
                       "confidence": float, "notes": str}, ...]
        }
    """

    def __init__(
        self,
        model: str = "gemini-3.1-flash-lite",
        timeout_s: float = 20.0,
        max_retries: int = 3,
        api_key_env: str = "GEMINI_API_KEY",
        prompt: str = HELMET_CHECK_PROMPT,
        estimated_cost_per_1k_tokens: float = 0.0,
    )->None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError( f"Missing API key: set the {api_key_env} environment variable" )
        self.client = OpenAI(api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/",)
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.prompt = prompt
        self.cost_per_1k = estimated_cost_per_1k_tokens

    def analyze(self, image_b64: str)-> dict:
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                return self._call_once(image_b64)
            except _RETRYABLE as exc:
                last_exc = exc
                wait = 2 ** attempt
                log.warning(
                    "LLM transient error (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1, self.max_retries, exc, wait,
                )
                time.sleep(wait)
            except (BadRequestError,) as exc:
                # Bad image / bad prompt — won't fix on retry.
                log.error("LLM rejected request: %s", exc)
                raise
        # All retries exhausted.
        assert last_exc is not None
        raise last_exc
    
    # -------------------------------------------------------------------
    # Internal: one call, parse + validate.
    # -------------------------------------------------------------------
    def _call_once(self, image_b64: str) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}",
                        "detail": "auto",   # let the API pick high/low detail
                    }},
                ],
            }],
            response_format={"type": "json_object"},
            timeout=self.timeout_s,
            temperature=0.0,   # deterministic — same frame, same verdict
        )

        # Log token usage for cost visibility.
        usage = getattr(response, "usage", None)
        if usage is not None:
            total = getattr(usage, "total_tokens", 0)
            cost = total / 1000.0 * self.cost_per_1k
            log.info(
                "LLM usage: prompt=%s completion=%s total=%s est_cost=$%.5f",
                getattr(usage, "prompt_tokens", "?"),
                getattr(usage, "completion_tokens", "?"),
                total, cost,
            )

        content = response.choices[0].message.content
        try:
            verdict = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned non-JSON: {content[:200]!r}") from exc

        self._validate(verdict)
        return verdict


    # -------------------------------------------------------------------
    # Internal: validate the parsed JSON shape.
    # -------------------------------------------------------------------
    def _validate(self, verdict: dict) -> None:
        if not isinstance(verdict, dict):
            raise ValueError("Verdict must be a dict")
        if not isinstance(verdict.get("unsafe"), bool):
            raise ValueError("Verdict.unsafe must be a bool")
        persons = verdict.get("persons")
        if not isinstance(persons, list):
            raise ValueError("Verdict.persons must be a list")
        for p in persons:
            if not isinstance(p, dict):
                raise ValueError("Each person must be a dict")
            if p.get("helmet") not in (True, False, None):
                raise ValueError(f"helmet must be bool|null, got {p.get('helmet')!r}")
            try:
                float(p.get("confidence", 0.0))
            except (TypeError, ValueError) as exc:
                raise ValueError("confidence must be numeric") from exc
    

    
