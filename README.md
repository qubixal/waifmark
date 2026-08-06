# WAIFMARK© 2 Benchmarking Suite

A benchmark testing **local agentic** and **roleplay/persona** capabilities of small (V)LLMs.

- **Agentic**: short-context toolcalling, strict JSON action output, file fetching/searching, calculations, lookup tables, safe shell usage, code error correction.
- **Persona**: multi-turn conversational context tracking, maintaining personality, safety traps, 中/Eng expression.

Max score is 100.00. Roleplay responses are scored by an LLM-as-a-judge with a triage stage that routes low-confidence verdicts to human auditing.

> The exact test bank is **proprietary** and is not part of this release. The tool ships with a dummy `data/test_bank.example.json` so it runs out of the box; drop your own `data/test_bank.json` in to benchmark.

## Install

Python 3.10+ recommended.

```bash
cd benchmark
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # or: pip install -e .
```

### Env for API keys

```bash
cp .env.example .env
```

Then set:

```bash
OPENROUTER_API_KEY=your_openrouter_key
HF_TOKEN=optional_hugging_face_token
```

## Run

```bash
streamlit run ui/app.py                # web UI (model search, run, leaderboard, audit)
```

or headless:

```bash
python main.py --config config.yaml --test-bank data/test_bank.json
```

`--mode pipeline` additionally starts/stops the serving backend (vLLM or llama.cpp for GGUF) around the run.

Before benchmarking, set `model` in `config.yaml` under `judges` to the judge model you want (e.g. `openrouter/deepseek-v3.1`), and point `model_under_test` at your model or GGUF file. Small models are sensitive to chat templates — if scores look unusually bad, verify your vLLM `--chat-template` and `config.yaml` `chat_template` settings.

## Scoring (v2)

This is a **breaking change from v1** — v2 scores are not directly comparable to v1 runs.

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

### Roleplay (LLM-as-judge)

Each transcript is scored 0–100 by the configured judge(s) against the persona rubric (`character_consistency`, `instruction_following`, `trap_resistance`, `conversational_quality`). The triage engine flags responses for human review on judge disagreement, boilerplate phrases ("As an AI language model..."), judge errors, and a seeded random spotcheck.

## Security notes

- The agentic sandbox is a **benchmark harness, not a security boundary**. Tasks run on your machine, and the model-under-test can invoke allowed shell commands (including `python3` by default). Set `agentic.allow_python3_tool: false` for untrusted test banks.
- Serving backends are spawned without a shell (argv-based), so model paths are never interpreted as shell commands.
- API keys live in `.env`, which is gitignored. Never commit `.env` or `data/test_bank.json`.

## Project layout

```
benchmark/
  core/          benchmark engine: sandbox, client, scoring, serving
  evaluation/    LLM judge + triage engine
  ui/            Streamlit control center
  data/          test banks (proprietary bank not committed)
  tests/         pytest suite (no network needed)
  config.yaml    run configuration (committed; no secrets)
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```
