# WAIFMARK© 2 Benchmarking Suite

A benchmark testing **local agentic** and **roleplay/persona** capabilities of small (V)LLMs.

- **Agentic**: short-context toolcalling, strict JSON action output, file fetching/searching, calculations, lookup tables, safe shell usage, code error correction.
- **Persona**: multi-turn conversational context tracking, maintaining personality, safety traps, 中/Eng expression.

Max score is 100.00. Roleplay responses are scored by an LLM-as-a-judge (v2) with a triage stage that routes low-confidence verdicts to human auditing.

> The exact test bank is **proprietary** and is not part of this release. The tool ships with a dummy `data/test_bank.example.json` so it runs out of the box; drop your own `data/test_bank.json` in to benchmark.

## Install & Boot (one command)

Python 3.10+ recommended.

```bash
cd benchmark
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # installs waifmark + FastAPI app
cp .env.example .env  # add OPENROUTER_API_KEY / HF_TOKEN
waifmark              # boots app at http://127.0.0.1:8001 (runs python -m api.run --host 127.0.0.1 --port 8001)
```

### Env for API keys (auto-created on boot if missing)

```bash
cp .env.example .env
# OPENROUTER_API_KEY=sk-or-…
# HF_TOKEN=hf_…
```

### Headless CLI

```bash
python main.py --config config.yaml --test-bank data/test_bank.json
```

`--mode pipeline` additionally starts/stops the serving backend (vLLM or llama.cpp for GGUF) around the run.

Before benchmarking, set `model` in `config.yaml` under `judges` to the judge model you want (e.g. `openrouter/deepseek-v3.1`), and point `model_under_test` at your model or GGUF file. Small models are sensitive to chat templates — if scores look unusually bad, verify your vLLM `--chat-template` and `config.yaml` `chat_template` settings.

## Scoring (v2)

V2 is a **complete refactor from v1**, so v2 scores are not directly comparable to v1.

### Agentic (deterministic)

Each step's tool call is validated against the task's tool list. Three components, weighted in `config.yaml`:

```
score_100 = 100 × ( tool_syntax_accuracy × 0.3
                  + goal_completion        × 0.5
                  + error_recovery         × 0.2 )
```

- `tool_syntax_accuracy` — fraction of steps whose action JSON was valid.
- `goal_completion` — blends final-answer hit rate (70%, configurable via `agentic.final_completion_weight`) with required-tool usage (30%).
- `error_recovery` — a failed step counts as recovered only when the next step succeeds and retries the *same* action.

### Roleplay (LLM-as-judge v2)

Each transcript is scored 0–100 by the configured judge(s) against the persona rubric (`character_consistency`, `instruction_following`, `trap_resistance`, `conversational_quality`). The v2 judge uses a calibrated, anchor-rich prompt with explicit Aura v2 persona, hard caps for `[DELETE]` / boilerplate, confidence-weighted aggregation, and a deterministic heuristic fallback when all judges fail. The triage engine flags responses for human review on judge disagreement, boilerplate phrases, **low confidence (<0.55)**, delete-marker flags, heuristic fallback, judge errors, and a seeded random spotcheck.

## Security notes

- The agentic sandbox is a **benchmark harness, not a security boundary**. Tasks run on your machine, and the model-under-test can invoke allowed shell commands (including `python3` by default). Set `agentic.allow_python3_tool: false` for untrusted test banks.
- Serving backends are spawned without a shell (argv-based) and via process groups (`start_new_session=True`) so `killpg` cleans workers. FastAPI binds `127.0.0.1` by default and enables CORS for the frontend.
- API keys live in `.env`, which is gitignored. Never commit `.env` or `data/test_bank.json`.

## Project layout

```
benchmark/
  core/          benchmark engine: sandbox, client, scoring, serving
  evaluation/    LLM judge v2 + triage engine v2
  api/           FastAPI backend (new app default)
  web/           Static frontend (Chart.js, vanilla JS — new app)
  data/          test banks (proprietary bank not committed)
  tests/         pytest suite (no network needed)
  config.yaml    run configuration (committed; no secrets)
  run.py         one-command boot helper
```