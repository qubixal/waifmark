"""Roleplay conversation runner with standardized multi-turn scripting."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from core.run_control import RunControl
from core.vllm_client import VLLMClient, VLLMConnectionError


class RoleplayArena:
    """Runs roleplay benchmark conversations against a target model."""

    def __init__(self, client: VLLMClient, config: dict[str, Any]) -> None:
        self.client = client
        self.config = config

    def run_task(
        self,
        task: dict[str, Any],
        control: RunControl | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        system_prompt = self._build_system_prompt(task)
        max_turns = int(task.get("max_turns", self.config["roleplay"].get("turns", 8)))
        turns = task.get("user_turns", [])[:max_turns]
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        transcript: list[dict[str, Any]] = []
        errors: list[str] = []

        for index, user_turn in enumerate(turns, start=1):
            if control:
                control.wait_if_paused()
            if progress_callback:
                progress_callback(
                    {
                        "module": "roleplay",
                        "task_id": task["id"],
                        "status": "in_progress",
                        "turn_index": index,
                        "turn_total": len(turns),
                    }
                )
            # Inject </think> to force-close any open thinking block (Qwen3.5 trick)
            # This helps thinking models stop reasoning and start the actual response
            user_content = user_turn
            # For known thinking models, append a hint to close thinking
            if "Qwen3.5" in self.client.model_name or "DeepSeek" in self.client.model_name or "Thinking" in self.client.model_name:
                user_content = user_turn + "\n\n</think>"
            messages.append({"role": "user", "content": user_content})
            assistant_text = ""
            assistant_raw = ""  # Keep raw thinking for logs
            metrics: dict[str, Any] = {}
            try:
                # For roleplay, explicitly disable thinking to prevent leaks
                # Qwen3.5 and DeepSeek thinking models leak "Thinking Process:" into the response
                result = self.client.chat(
                    messages,
                    temperature=float(task.get("temperature", self.config["roleplay"]["temperature"])),
                    max_tokens=int(task.get("max_tokens", self.config["roleplay"]["max_tokens"])),
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                assistant_raw = result.raw_content if hasattr(result, 'raw_content') and result.raw_content else result.content
                assistant_text = result.content  # Already stripped via vllm_client
                metrics = result.metrics
                # Keep thinking for record but use stripped for conversation
                messages.append({"role": "assistant", "content": assistant_text})
            except VLLMConnectionError as exc:
                assistant_text = f"[vLLM error: {exc}]"
                metrics = {}
                errors.append(str(exc))
                messages.append({"role": "assistant", "content": assistant_text})

            transcript.append(
                {
                    "turn": index,
                    "user": user_turn,
                    "assistant": assistant_text,
                    "assistant_raw": assistant_raw,
                    "thinking_tokens": metrics.get("thinking_tokens", 0) if isinstance(metrics, dict) else 0,
                    "is_trap_turn": index in task.get("trap_turn_indices", []),
                    "metrics": metrics,
                }
            )

        combined_response = "\n".join(item["assistant"] for item in transcript)
        combined_raw = "\n".join(item.get("assistant_raw", item["assistant"]) for item in transcript)
        # Track thinking stats for analysis
        total_thinking = sum(item.get("thinking_tokens", 0) for item in transcript)
        return {
            "task_id": task["id"],
            "task_type": "roleplay",
            "character_name": task["character_name"],
            "transcript": transcript,
            "combined_response": combined_response,
            "combined_raw": combined_raw,
            "total_thinking_tokens": total_thinking,
            "errors": errors,
        }

    def _build_system_prompt(self, task: dict[str, Any]) -> str:
        # Prefer the task's own character prompt so test banks can define
        # per-task personas. Falls back to the shared config persona (Aura)
        # for banks that rely on the global default.
        task_prompt = task.get("character_prompt", "").strip()
        if task_prompt:
            # Append thinking suppression for thinking models
            thinking_guard = "\n\nIMPORTANT: Do NOT show your thinking process, reasoning, or 'Thinking Process:' sections. Only output the final in-character response in Aura's voice."
            return task_prompt + thinking_guard
        base_prompt = self.config.get("roleplay", {}).get("system_prompt", "")
        if not base_prompt:
            # Fallback: build from task fields (legacy)
            base_prompt = f"You are roleplaying as {task['character_name']}."
        thinking_guard = "\n\nIMPORTANT: Do NOT show your thinking process, reasoning, or 'Thinking Process:' sections. Only output the final in-character response."
        return base_prompt + thinking_guard
