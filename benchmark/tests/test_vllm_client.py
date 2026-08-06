"""Tests for the OpenAI-compatible client helpers."""

from __future__ import annotations

import pytest

from core.vllm_client import (
    ChatResult,
    InvalidJSONResponseError,
    VLLMClient,
    VLLMConnectionError,
    estimate_token_count,
    extract_json_object,
    strip_thinking_content,
)


class FakeResponse:
    def __init__(self, content: str, usage: dict | None = None) -> None:
        self.body = {
            "choices": [{"message": {"content": content}}],
            "usage": usage or {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        }


class FakeClient(VLLMClient):
    """VLLMClient whose chat() returns canned responses instead of HTTP."""

    def __init__(self, responses, retries: int = 1):
        self._responses = list(responses)
        self._calls = 0
        super().__init__(
            {"base_url": "http://localhost:8000/v1", "model": "test", "retries": retries, "provider": "vllm"}
        )

    def chat(self, messages, **kwargs):
        self._calls += 1
        if not self._responses:
            raise VLLMConnectionError("endpoint unreachable")
        content = self._responses.pop(0)
        if isinstance(content, Exception):
            raise content
        result = FakeResponse(content)
        return self._wrap_result(result)

    def _wrap_result(self, response: FakeResponse) -> ChatResult:
        body = response.body
        choice = body["choices"][0]["message"]
        return ChatResult(content=choice["content"], raw_response=body, usage=body["usage"], metrics={})


def test_estimate_token_count_english():
    assert estimate_token_count("hello world") >= 1
    assert estimate_token_count("") == 0


def test_extract_json_object_from_fenced_text():
    text = 'Here you go:\n```json\n{"a": 1}\n```\nbye'
    assert extract_json_object(text) == {"a": 1}


def test_extract_json_object_first_object_wins():
    text = '{"a": 1} trailing {"b": 2}'
    assert extract_json_object(text) == {"a": 1}


def test_extract_json_object_raises_when_missing():
    with pytest.raises(InvalidJSONResponseError):
        extract_json_object("no json here at all")


def test_strip_thinking_blocks():
    stripped = strip_thinking_content("hello <think>secret plan</think> world")
    assert "secret" not in stripped and "hello" in stripped and "world" in stripped
    assert strip_thinking_content("<thinking>secret</thinking>answer") == "answer"
    assert strip_thinking_content("no tags") == "no tags"


def test_chat_json_parses_clean_content():
    client = FakeClient(['{"thought": "t", "action": "final_answer", "args": {}, "final_answer": "42"}'])
    result = client.chat_json([])
    assert result["final_answer"] == "42"
    assert client._calls == 1


def test_chat_json_recovers_from_broken_json_then_repairs():
    client = FakeClient(['{"thought": "t"', '{"action": "final_answer", "args": {}, "final_answer": "ok"}'])
    result = client.chat_json([], repair_attempts=2)
    assert result["action"] == "final_answer"


def test_chat_json_raises_after_repair_attempts_exhausted():
    client = FakeClient(["not json at all", "still not json", "nope"])
    with pytest.raises(InvalidJSONResponseError):
        client.chat_json([], repair_attempts=1)
    assert client._calls == 2


def test_chat_json_repair_attempts_zero_means_no_repair():
    client = FakeClient(["not json at all"])
    with pytest.raises(InvalidJSONResponseError):
        client.chat_json([], repair_attempts=0)
    assert client._calls == 1


def test_chat_raises_connection_error_when_responses_exhausted():
    client = FakeClient([])
    with pytest.raises(VLLMConnectionError):
        client.chat([])
