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
            messages.append({"role": "user", "content": user_turn})
            assistant_text = ""
            metrics: dict[str, Any] = {}
            try:
                result = self.client.chat(
                    messages,
                    temperature=float(task.get("temperature", self.config["roleplay"]["temperature"])),
                    max_tokens=int(task.get("max_tokens", self.config["roleplay"]["max_tokens"])),
                )
                assistant_text = result.content
                metrics = result.metrics
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
                    "is_trap_turn": index in task.get("trap_turn_indices", []),
                    "metrics": metrics,
                }
            )

        combined_response = "\n".join(item["assistant"] for item in transcript)
        return {
            "task_id": task["id"],
            "task_type": "roleplay",
            "character_name": task["character_name"],
            "transcript": transcript,
            "combined_response": combined_response,
            "errors": errors,
        }

    def _build_system_prompt(self, task: dict[str, Any]) -> str:
        # Prefer the task's own character prompt so test banks can define
        # per-task personas. Falls back to the shared config persona (Aura)
        # for banks that rely on the global default.
        task_prompt = task.get("character_prompt", "").strip()
        if task_prompt:
            return task_prompt
        base_prompt = self.config.get("roleplay", {}).get("system_prompt", "")
        if not base_prompt:
            # Fallback: build from task fields (legacy)
            base_prompt = f"You are roleplaying as {task['character_name']}."
        return base_prompt
