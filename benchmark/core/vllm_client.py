"""OpenAI-compatible vLLM client with JSON enforcement helpers."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

LOGGER = logging.getLogger(__name__)


class VLLMError(RuntimeError):
    """Base error for vLLM communication failures."""


class VLLMConnectionError(VLLMError):
    """Raised when the local vLLM endpoint is unavailable."""


class InvalidJSONResponseError(VLLMError):
    """Raised when a model response cannot be coerced into valid JSON."""


@dataclass
class ChatResult:
    """Normalized chat completion result."""

    content: str
    raw_response: dict[str, Any]
    usage: dict[str, Any]
    metrics: dict[str, Any]
    raw_content: str = ""
    stripped_content: str = ""


def estimate_token_count(text: str) -> int:
    """Cheap fallback when a provider omits completion token usage.

    Uses character-based estimation (~4 chars per token) which works
    reasonably for both English and CJK text, unlike word-based splitting.
    """
    if not text:
        return 0
    return max(1, round(len(text) / 4))


def strip_thinking_content(content: str) -> str:
    """Remove thinking/reasoning blocks from model output.

    Handles <think>...</think> tags and plain 'Thinking Process:' leaks used by thinking-only models
    (e.g. Qwen3, DeepSeek-R1, Qwen3.5) so the actual response is extracted.
    If no thinking tags are found, returns the content unchanged.
    """
    import re
    # Remove <think>...</think> blocks (case-insensitive, multiline)
    stripped = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)
    # Also handle variants: <thinking>, <reason>, <thought>
    stripped = re.sub(r"<thinking>.*?</thinking>", "", stripped, flags=re.DOTALL | re.IGNORECASE)
    stripped = re.sub(r"<reason>.*?</reason>", "", stripped, flags=re.DOTALL | re.IGNORECASE)
    stripped = re.sub(r"<thought>.*?</thought>", "", stripped, flags=re.DOTALL | re.IGNORECASE)
    # Handle plain "Thinking Process:" leaks (Qwen3.5, DeepSeek) — remove up to the actual response
    # These models often leak: "Thinking Process:\n1. Analyze...\n..." without closing tag, or the entire response is thinking
    # Handle both at start and anywhere in the content
    if re.search(r"Thinking Process:", stripped, re.IGNORECASE):
        # If the content is very long and contains Thinking Process, it's likelythinking leak
        if len(stripped) > 800 and "Analyze" in stripped[:1500]:
            # Try to find where actual response starts — look for markers after thinking
            # Common markers: "Response:", "Answer:", "Aura:", "Sensei", "---", or a short conversational start
            parts = re.split(r"\n\s*(?:Constraints:|Response:|Answer:|Aura:|Sensei|---)\s*\n", stripped, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) > 1 and len(parts[1].strip()) > 20 and len(parts[1].strip()) < len(stripped) * 0.7:
                stripped = parts[1]
            else:
                # Fallback: if still looks like thinking, try to extract the last part that looks like a real response
                # Real responses are usually <300 chars and conversational, thinking is >800 chars
                # Find the last occurrence of a potential response start
                # Look for a short, conversational snippet after the thinking
                match = re.search(r"(Sensei[~!].*|Hehe.*|Oh, Sensei.*|Aww.*|Hmm.*)", stripped, re.DOTALL | re.IGNORECASE)
                if match and len(match.group(1).strip()) > 20 and len(match.group(1).strip()) < 800:
                    stripped = match.group(1).strip()
                elif len(stripped) > 1000:
                    # If still very long and looks like thinking, truncate to last 500 chars which is likely the real response
                    # But only if the last 500 chars don't contain "Analyze" or "Thinking Process"
                    last_part = stripped[-800:].strip()
                    if "Analyze" not in last_part and "Thinking Process" not in last_part and len(last_part) > 20:
                        stripped = last_part
    # Also handle "Reasoning:" plain leaks anywhere
    stripped = re.sub(r"Thinking Process:.*?(?=\n[A-Z][a-z]+,|\nHai |Sensei|~|💕|😊|$)", "", stripped, flags=re.DOTALL | re.IGNORECASE)
    stripped = re.sub(r"^\s*Reasoning:.*?\n\s*\n", "", stripped, flags=re.DOTALL | re.IGNORECASE)
    stripped = re.sub(r"Reasoning:.*?\n\s*\n", "", stripped, flags=re.DOTALL | re.IGNORECASE)
    return stripped.strip()


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first valid JSON object from free-form text."""
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise InvalidJSONResponseError("No valid JSON object found in model output.")


class VLLMClient:
    """Small wrapper around a vLLM OpenAI-compatible endpoint."""

    def __init__(self, config: dict[str, Any], model_name: str | None = None) -> None:
        self.config = config
        self.model_name = model_name or config.get("model") or config.get("name")
        self.base_url = config["base_url"].rstrip("/")
        self.api_key = os.environ.get(config.get("api_key_env", ""), config.get("api_key", "EMPTY"))
        self.timeout_seconds = int(config.get("timeout_seconds", 60))
        self.retries = int(config.get("retries", 3))
        self.default_temperature = float(config.get("temperature", 0.0))
        self.default_max_tokens = int(config.get("max_tokens", 2048))
        self.chat_template = config.get("chat_template")
        self.add_generation_prompt = bool(config.get("add_generation_prompt", True))
        self.provider = config.get("provider", "vllm")
        self.supports_guided_json = bool(config.get("supports_guided_json", self.provider == "vllm"))
        self.include_vllm_template_fields = bool(config.get("include_vllm_template_fields", self.provider == "vllm"))
        self.last_chat_result: ChatResult | None = None
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def chat(
        self,
        messages: Iterable[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        response_schema: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": list(messages),
            "temperature": self.default_temperature if temperature is None else temperature,
            "max_tokens": self.default_max_tokens if max_tokens is None else max_tokens,
        }
        if self.include_vllm_template_fields:
            if self.chat_template:
                payload["chat_template"] = self.chat_template
            payload["add_generation_prompt"] = self.add_generation_prompt
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if response_schema and self.supports_guided_json:
            payload["guided_json"] = response_schema
        if extra_body:
            for key, value in extra_body.items():
                if key not in payload:
                    payload[key] = value
                elif isinstance(payload[key], dict) and isinstance(value, dict):
                    payload[key].update(value)

        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                started_at = time.perf_counter()
                request = Request(
                    f"{self.base_url}/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers=self.headers,
                    method="POST",
                )
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    body = json.loads(response.read().decode("utf-8"))
                elapsed_seconds = max(time.perf_counter() - started_at, 0.001)
                choice = body["choices"][0]["message"]
                content = choice.get("content", "")
                # Fallback for thinking/reasoning models: check reasoning_content and reasoning fields
                # (llama.cpp uses reasoning_content, some OpenRouter models use reasoning)
                if not content or not str(content).strip():
                    content = choice.get("reasoning_content", "") or choice.get("reasoning", "") or ""
                # Some providers put reasoning in separate field even when content exists; combine if content empty
                if isinstance(content, list):
                    content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
                raw_content = str(content).strip()
                # If still empty but reasoning fields exist, use them
                if not raw_content:
                    rc = choice.get("reasoning_content", "") or choice.get("reasoning", "")
                    if rc:
                        raw_content = str(rc).strip()
                stripped_content = strip_thinking_content(raw_content)
                content = stripped_content
                usage = body.get("usage", {})
                # Raw tokens include thinking; stripped tokens exclude it.
                # Prefer provider usage for raw, but estimate both for fallback.
                raw_tokens_est = estimate_token_count(raw_content)
                stripped_tokens_est = estimate_token_count(stripped_content)
                thinking_tokens_est = max(0, raw_tokens_est - stripped_tokens_est)
                # Provider completion_tokens (if present) is authoritative for raw.
                completion_tokens = int(usage.get("completion_tokens") or raw_tokens_est)
                # Stripped tokens for judge/verbosity (not penalized for thinking)
                stripped_completion_tokens = int(stripped_tokens_est)
                # Keep raw for latency accounting; stripped for content.
                prompt_tokens = int(usage.get("prompt_tokens") or 0)
                total_tokens = int(usage.get("total_tokens") or prompt_tokens + completion_tokens)
                tokens_per_second = completion_tokens / elapsed_seconds if completion_tokens else 0.0
                # tokens_per_second based on raw (includes thinking) so time includes thinking
                stripped_tps = stripped_completion_tokens / elapsed_seconds if stripped_completion_tokens else 0.0
                result = ChatResult(
                    content=content,
                    raw_response=body,
                    usage=usage,
                    metrics={
                        "completion_tokens": completion_tokens,
                        "prompt_tokens": prompt_tokens,
                        "total_tokens": total_tokens,
                        "tokens_per_second": round(tokens_per_second, 3),
                        "time_seconds": round(elapsed_seconds, 3),
                        "latency_seconds": round(elapsed_seconds, 3),
                        "stripped_completion_tokens": stripped_completion_tokens,
                        "thinking_tokens": thinking_tokens_est,
                        "stripped_tokens_per_second": round(stripped_tps, 3),
                    },
                    raw_content=raw_content,
                    stripped_content=stripped_content,
                )
                self.last_chat_result = result
                return result
            except (HTTPError, URLError, TimeoutError, KeyError, ValueError, JSONDecodeError) as exc:
                last_error = exc
                LOGGER.warning("vLLM request failed on attempt %s/%s: %s", attempt, self.retries, exc)
                if attempt < self.retries:
                    time.sleep(min(2 ** attempt, 30))
        raise VLLMConnectionError(f"Failed to reach vLLM endpoint at {self.base_url}: {last_error}") from last_error

    def chat_json(
        self,
        messages: Iterable[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_schema: dict[str, Any] | None = None,
        repair_attempts: int = 1,
    ) -> dict[str, Any]:
        result = self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
            response_schema=response_schema,
        )
        for attempt in range(repair_attempts + 1):
            try:
                return json.loads(result.content)
            except JSONDecodeError:
                pass

            try:
                return extract_json_object(result.content)
            except InvalidJSONResponseError:
                pass

            if attempt >= repair_attempts:
                break

            repair_messages = list(messages) + [
                {
                    "role": "system",
                    "content": (
                        "Your previous response was not valid JSON. "
                        "Return ONLY a valid JSON object that preserves the original meaning. "
                        "Do not add markdown or commentary."
                    ),
                },
                {"role": "user", "content": f"Here was your invalid response — please fix it:\n\n{result.content}"},
            ]
            result = self.chat(repair_messages, temperature=0.0, max_tokens=max_tokens or 400, json_mode=True)

        raise InvalidJSONResponseError(
            f"Model returned invalid JSON after {repair_attempts} repair attempt(s): {result.content}"
        )
