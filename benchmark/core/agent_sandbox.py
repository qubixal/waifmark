"""Short-context agentic benchmark execution sandbox."""

from __future__ import annotations

import ast
import json
import math
import shlex
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.run_control import RunControl
from core.vllm_client import InvalidJSONResponseError, VLLMClient, VLLMConnectionError

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"},
        "action": {"type": "string"},
        "args": {"type": "object"},
        "final_answer": {"type": "string"},
    },
    "required": ["thought", "action", "args", "final_answer"],
}


SAFE_SHELL_COMMANDS = {"pwd", "ls", "cat", "echo", "wc", "head", "tail", "python3", "awk", "sed", "grep", "sort", "uniq", "tr", "cut", "paste"}


@dataclass
class AgentStepResult:
    step_index: int
    model_action: dict[str, Any]
    syntax_valid: bool
    tool_output: str
    error: str | None
    metrics: dict[str, Any]


class AgentSandbox:
    """Executes short-context agent tasks with a small safe tool surface."""

    def __init__(self, client: VLLMClient, config: dict[str, Any]) -> None:
        self.client = client
        self.config = config
        self._allow_python3_tool = bool(config.get("agentic", {}).get("allow_python3_tool", True))

    def run_task(
        self,
        task: dict[str, Any],
        control: RunControl | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        # Use task-specific max_steps if provided, otherwise use config default, capped at 10
        max_steps = min(int(task.get("max_steps", self.config["run"]["max_agent_steps"])), 10)
        history: list[AgentStepResult] = []
        tool_calls: list[str] = []
        final_answer = ""

        with tempfile.TemporaryDirectory(prefix="benchmark-agent-") as tmpdir:
            workspace = Path(tmpdir)
            for relative_path, content in task.get("workspace_files", {}).items():
                file_path = workspace / relative_path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content, encoding="utf-8")

            for step_index in range(1, max_steps + 1):
                if control:
                    control.wait_if_paused()
                if progress_callback:
                    progress_callback(
                        {
                            "module": "agentic",
                            "task_id": task["id"],
                            "status": "in_progress",
                            "step_index": step_index,
                            "step_total": max_steps,
                        }
                    )
                prompt_messages = self._build_messages(task, history, workspace)
                try:
                    action = self.client.chat_json(
                        prompt_messages,
                        temperature=self.client.default_temperature,
                        max_tokens=self.client.default_max_tokens,
                        response_schema=ACTION_SCHEMA if self.config["agentic"].get("enforce_json") else None,
                        repair_attempts=int(self.config["agentic"].get("json_repair_attempts", 1)),
                    )
                except (InvalidJSONResponseError, VLLMConnectionError) as exc:
                    history.append(
                        AgentStepResult(
                            step_index=step_index,
                            model_action={"raw_error": str(exc)},
                            syntax_valid=False,
                            tool_output="",
                            error=str(exc),
                            metrics={},
                        )
                    )
                    break

                syntax_valid, validation_error = self._validate_action(action, task)
                if not syntax_valid:
                    history.append(
                        AgentStepResult(
                            step_index=step_index,
                            model_action=action,
                            syntax_valid=False,
                            tool_output="",
                            error=validation_error,
                            metrics=self.client.last_chat_result.metrics if self.client.last_chat_result else {},
                        )
                    )
                    continue

                if action["action"] == "final_answer":
                    final_answer = str(action.get("final_answer", "")).strip()
                    history.append(
                        AgentStepResult(
                            step_index=step_index,
                            model_action=action,
                            syntax_valid=True,
                            tool_output="Submitted final answer.",
                            error=None,
                            metrics=self.client.last_chat_result.metrics if self.client.last_chat_result else {},
                        )
                    )
                    break

                tool_calls.append(action["action"])
                tool_output, tool_error = self._execute_tool(action, task, workspace)
                history.append(
                    AgentStepResult(
                        step_index=step_index,
                        model_action=action,
                        syntax_valid=True,
                        tool_output=tool_output,
                        error=tool_error,
                        metrics=self.client.last_chat_result.metrics if self.client.last_chat_result else {},
                    )
                )

        metrics = self._score_task(task, history, final_answer, tool_calls)
        return {
            "task_id": task["id"],
            "task_type": "agentic",
            "goal": task["goal"],
            "final_answer": final_answer,
            "steps": [self._serialize_step(item) for item in history],
            "tool_calls": tool_calls,
            "metrics": metrics,
            "requires_review_hint": metrics["goal_completion"] < 0.5 or metrics["tool_syntax_accuracy"] < 0.5,
        }

    def _build_messages(
        self,
        task: dict[str, Any],
        history: list[AgentStepResult],
        workspace: Path,
    ) -> list[dict[str, str]]:
        tool_lines = []
        for tool in task.get("tools", []):
            tool_lines.append(f"- {tool['name']}: {tool['description']}")
        history_blob = []
        for item in history:
            history_blob.append(
                {
                    "step": item.step_index,
                    "action": item.model_action,
                    "syntax_valid": item.syntax_valid,
                    "tool_output": item.tool_output,
                    "error": item.error,
                }
            )

        # Build per-tool examples showing exact arg key names
        tool_examples = []
        for tool in task.get("tools", []):
            name = tool["name"]
            if name == "shell":
                tool_examples.append(
                    '  {"thought": "...", "action": "shell", '
                    '"args": {"command": "cat file.txt"}, "final_answer": null}'
                )
            elif name == "search_text":
                tool_examples.append(
                    '  {"thought": "...", "action": "search_text", '
                    '"args": {"file": "notes.txt", "query": "keyword"}, "final_answer": null}'
                )
            elif name == "calculator":
                tool_examples.append(
                    '  {"thought": "...", "action": "calculator", '
                    '"args": {"expression": "1500 - 829"}, "final_answer": null}'
                )
            elif name == "lookup":
                tool_examples.append(
                    '  {"thought": "...", "action": "lookup", '
                    '"args": {"key": "some_key"}, "final_answer": null}'
                )
            elif name == "final_answer":
                tool_examples.append(
                    '  {"thought": "...", "action": "final_answer", '
                    '"args": {}, "final_answer": "The answer is 42."}'
                )
            else:
                tool_examples.append(f'  {{"thought": "...", "action": "{name}", "args": {{}}, "final_answer": null}}')

        system_prompt = (
            "You are an autonomous benchmark agent operating in a short-context tool sandbox. "
            "Return ONLY a JSON object with exactly these keys: thought, action, args, final_answer. "
            "DO NOT include markdown fences, code blocks, or any text outside the JSON. "
            "\n\nTool call format (args keys are EXACT — use the key names shown):\n"
            + "\n".join(tool_examples) +
            "\n\nRules:\n"
            "- Use action='final_answer' with a non-empty final_answer when you have the answer.\n"
            "- If a tool fails, read the error and try a different approach.\n"
            "- Keep thought brief (1-2 sentences).\n"
            "- Do not invent tool names or arg keys."
        )
        user_prompt = {
            "goal": task["goal"],
            "max_steps": task.get("max_steps", self.config["run"]["max_agent_steps"]),
            "workspace_root": str(workspace),
            "available_tools": tool_lines,
            "history": history_blob,
            "ground_rules": [
                "Do not invent tool names or arg key names.",
                "Do not include markdown fences.",
                "Keep thought brief.",
                "When you have the answer, call final_answer immediately — do not keep exploring.",
            ],
        }
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
        ]

    def _validate_action(self, action: dict[str, Any], task: dict[str, Any]) -> tuple[bool, str | None]:
        required_keys = {"thought", "action", "args"}
        if not isinstance(action, dict) or not required_keys.issubset(action.keys()):
            return False, "Action JSON is missing required keys."
        # final_answer is optional for non-final actions; inject null if missing
        if "final_answer" not in action:
            action["final_answer"] = None
        if not isinstance(action.get("args"), dict):
            return False, "Action args must be a JSON object."
        valid_actions = {tool["name"] for tool in task.get("tools", [])}
        valid_actions.add("final_answer")
        if action["action"] not in valid_actions:
            return False, f"Unknown action '{action['action']}'."
        if action["action"] == "final_answer" and not str(action.get("final_answer", "")).strip():
            return False, "Final answer action requires non-empty final_answer."
        return True, None

    def _execute_tool(self, action: dict[str, Any], task: dict[str, Any], workspace: Path) -> tuple[str, str | None]:
        tool_name = action["action"]
        args = action.get("args", {})
        try:
            if tool_name == "shell":
                return self._run_shell(args, workspace), None
            if tool_name == "search_text":
                return self._search_text(args, workspace), None
            if tool_name == "calculator":
                return self._calculate(args), None
            if tool_name == "lookup":
                return self._lookup(args, task), None
            return "", f"Unsupported tool implementation for '{tool_name}'."
        except Exception as exc:
            return "", str(exc)

    def _run_shell(self, args: dict[str, Any], workspace: Path) -> str:
        command = str(args.get("command", "")).strip()
        if not command:
            raise ValueError("shell tool requires a command.")
        parts = shlex.split(command)
        if not parts:
            raise ValueError("shell command is empty.")
        if parts[0] not in SAFE_SHELL_COMMANDS:
            raise ValueError(f"Command '{parts[0]}' is not allowed in sandbox.")
        if parts[0] == "python3" and not self._allow_python3_tool:
            raise ValueError(
                "python3 is disabled for this run (agentic.allow_python3_tool=false). "
                "Use one of the other allowed commands."
            )
        # Reject arguments that attempt path traversal or absolute paths
        for part in parts[1:]:
            if ".." in part:
                raise ValueError("Shell arguments must not contain '..'.")
            if part.startswith("/") or part.startswith("~"):
                raise ValueError("Shell arguments must not be absolute paths.")
        completed = subprocess.run(
            parts,
            cwd=workspace,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        return json.dumps(
            {
                "returncode": completed.returncode,
                "stdout": stdout,
                "stderr": stderr,
            },
            ensure_ascii=False,
        )

    def _search_text(self, args: dict[str, Any], workspace: Path) -> str:
        file_name = args.get("file")
        query = str(args.get("query", "")).strip().lower()
        if not file_name or not query:
            raise ValueError("search_text requires file and query arguments.")
        target = (workspace / str(file_name)).resolve()
        if not target.is_relative_to(workspace.resolve()):
            raise ValueError("Path traversal detected — file must be inside the workspace.")
        if not target.exists():
            raise FileNotFoundError(f"File not found: {target.name}")
        matches = []
        for line_number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
            if query in line.lower():
                matches.append({"line": line_number, "text": line})
        return json.dumps({"matches": matches}, ensure_ascii=False)

    def _lookup(self, args: dict[str, Any], task: dict[str, Any]) -> str:
        key = str(args.get("key", "")).strip()
        lookup_table = task.get("lookup_table", {})

        # Allow models to discover available keys
        if key in ("__keys__", "list", "keys"):
            available = sorted(lookup_table.keys())
            return json.dumps({"available_keys": available}, ensure_ascii=False)

        if key not in lookup_table:
            available = ", ".join(sorted(lookup_table.keys()))
            raise KeyError(f"Unknown lookup key: '{key}'. Available keys: {available}")
        return json.dumps({"key": key, "value": lookup_table[key]}, ensure_ascii=False)

    def _calculate(self, args: dict[str, Any]) -> str:
        expression = str(args.get("expression", "")).strip()
        if not expression:
            raise ValueError("calculator requires an expression.")
        value = self._safe_eval(expression)
        return json.dumps({"expression": expression, "value": value}, ensure_ascii=False)

    def _safe_eval(self, expression: str) -> float:
        allowed_nodes = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Mod,
            ast.USub,
            ast.UAdd,
            ast.Constant,
            ast.Load,
        )
        tree = ast.parse(expression, mode="eval")
        if any(not isinstance(node, allowed_nodes) for node in ast.walk(tree)):
            raise ValueError("calculator expression contains unsupported syntax.")
        try:
            value = eval(compile(tree, "<calculator>", "eval"), {"__builtins__": {}}, {})
        except ArithmeticError as exc:
            raise ValueError("calculator expression returned an invalid numeric result.") from exc
        if isinstance(value, complex) or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
            raise ValueError("calculator expression returned invalid numeric output.")
        return float(value)

    def _score_task(
        self,
        task: dict[str, Any],
        history: list[AgentStepResult],
        final_answer: str,
        tool_calls: list[str],
    ) -> dict[str, Any]:
        valid_steps = sum(1 for step in history if step.syntax_valid)
        tool_syntax_accuracy = round(valid_steps / max(len(history), 1), 3)

        final_targets = [item.lower() for item in task.get("ground_truth", {}).get("final_answer_contains", [])]
        required_tools = task.get("ground_truth", {}).get("required_tool_calls", [])

        # Partial credit for tool usage
        if required_tools:
            matched_tools = sum(1 for t in required_tools if t in tool_calls)
            tool_score = matched_tools / len(required_tools)
        else:
            tool_score = 1.0

        # Partial credit for final answer
        if final_targets:
            matched_targets = sum(1 for t in final_targets if t in final_answer.lower())
            final_score = matched_targets / len(final_targets) if final_answer.strip() else 0.0
        else:
            final_score = 1.0 if final_answer.strip() else 0.0

        # Tiered scoring (vNext): empty final => 0, tool-only partial, full+correct =>100.
        # Complexity-scaled tool credit: 1 tool -> 35 max, 2 tools -> 40 max.
        total_errors = sum(1 for step in history if step.error)
        recovered_errors = 0
        for index, step in enumerate(history[:-1]):
            if not step.error:
                continue
            nxt = history[index + 1]
            if not nxt.error and nxt.syntax_valid and nxt.model_action.get("action") == step.model_action.get("action"):
                recovered_errors += 1
        error_recovery = round(recovered_errors / total_errors, 3) if total_errors else 1.0

        # If nothing was written (no final_answer), score is 0 regardless of tool calls.
        if not final_answer.strip():
            return {
                "tool_syntax_accuracy": tool_syntax_accuracy,
                "goal_completion": 0.0,
                "error_recovery": error_recovery,
                "score_100": 0.0,
                "total_errors": total_errors,
                "tool_score": round(tool_score, 3),
                "final_score": round(final_score, 3),
            }

        # Complexity factor: more required tools / longer horizon => slightly higher tool-only ceiling.
        # 1 required -> 35, 2 -> 40, 3+ -> 45; plus max_steps bonus up to +5.
        base_tool_ceiling = 30 + len(required_tools) * 5 if required_tools else 35
        # clamp 30-45
        base_tool_ceiling = max(30, min(45, base_tool_ceiling))
        max_steps = int(task.get("max_steps", self.config["run"]["max_agent_steps"]))
        # up to +5 for horizon
        horizon_bonus = min(5, max(0, (max_steps - 6) * 1.0))
        tool_ceiling = base_tool_ceiling + horizon_bonus  # 30-50 range

        final_weight = float(self.config.get("agentic", {}).get("final_completion_weight", 0.7))
        tool_weight = 1.0 - final_weight

        # Determine overall via tiers
        if final_score == 1.0 and tool_score == 1.0:
            overall = 100.0
            goal_completion = 1.0
        elif final_score == 1.0:
            # correct answer but missing some required tool
            overall = round(70 + tool_score * 15, 2)  # 70-85
            goal_completion = round(final_score * final_weight + tool_score * tool_weight, 3)
        elif final_score > 0:
            # partial final: diminishing credit (e.g. 1/2 => not half)
            overall = round(tool_score * tool_ceiling * 0.7 + final_score * 40, 2)
            # cap partial at 65 if not all targets hit
            overall = min(overall, 65.0)
            goal_completion = round(final_score * final_weight + tool_score * tool_weight, 3)
        else:
            # wrong answer but final was submitted: tool-only credit
            overall = round(tool_score * tool_ceiling, 2)
            goal_completion = round(tool_score * tool_weight, 3)  # final 0

        # Preserve legacy weighted path for goal_completion when final empty is already handled.
        # For non-empty cases, also compute legacy goal for diagnostics but override with tiered overall.
        # Keep overall within 0-100
        overall = round(max(0.0, min(100.0, overall)), 2)

        return {
            "tool_syntax_accuracy": tool_syntax_accuracy,
            "goal_completion": round(goal_completion, 3),
            "error_recovery": error_recovery,
            "score_100": overall,
            "total_errors": total_errors,
            "tool_score": round(tool_score, 3),
            "final_score": round(final_score, 3),
        }

    def _serialize_step(self, item: AgentStepResult) -> dict[str, Any]:
        return {
            "step_index": item.step_index,
            "model_action": item.model_action,
            "syntax_valid": item.syntax_valid,
            "tool_output": item.tool_output,
            "error": item.error,
            "metrics": item.metrics,
        }
