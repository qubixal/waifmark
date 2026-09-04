# WAIFMARK© 2 <small>Benchmarking Suite</small>

Waifmark 2 is a benchmark testing **local agentic** and **roleplay/persona** capabilities of small (V)LLMs.

- **Agentic**: short-context toolcalling, strict JSON action output, file fetching/searching, calculations, lookup tables, safe shell usage, code error correction.
- **Persona**: multi-turn conversational context tracking, maintaining personality, safety traps, 中/Eng expression.

Max score is 100.00. Roleplay responses are scored by an LLM-as-a-judge (v2) with a triage stage that routes low-confidence verdicts to human auditing.

> The exact test bank is **proprietary** and is not part of this release. The tool ships with a dummy `data/test_bank.example.json` so it runs out of the box; drop your own `data/test_bank.json` in to benchmark.

## Install & Boot

Python 3.10+ is recommended for installation.

```bash
cd benchmark
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # install reqs
cp .env.example .env                   # add OPENROUTER_API_KEY / HF_TOKEN api keys
waifmark                               # ← boots Frontend app at http://127.0.0.1:8001
```

To shut down:
```bash
waifmark --stop
```

The run and terminate commands have been simplified to "waifmark" for convience.

### Env for API keys

```bash
cp .env.example .env
# then either nano into the file or open in a editor to enter your
# OPENROUTER_API_KEY=sk-or-…, HF_TOKEN=hf_…
```

### Headless CLI (No frontend)

If for example you would like to run consecutive benchmarks, or don't want to use the frontend UI:
```bash
python main.py --config config.yaml --test-bank data/test_bank.json
```

use `--mode pipeline` to start/stop the backend (vLLM, llama.cpp) serving the model.

### Scoring Judge
Before you benchmark, the judge needs to be set. You can get an api key from [Openrouter](https://openrouter.ai), then set `model` in `config.yaml` under `judges` to the judge model you want (e.g. `openrouter/free`), and point `model_under_test` at your model or GGUF file.

> Note: Small models are sometimes sensitive to chat templates. If your benchmark scores look unusually bad, verify your vLLM `--chat-template` and `config.yaml` `chat_template`.

## Scoring (v2)

**v2 changed a lot of things from v1**, and therefore are not directly comparable.

### Part 1 (1/3): Agentic

Involves working in a sandbox; each step's tool call is checked against the task's tool list.
There are three score components, weighed in `config.yaml`:

```
score_100 = 100 × ( tool_syntax_accuracy × 0.3
                  + goal_completion        × 0.5
                  + error_recovery         × 0.2 )
```

- `tool_syntax_accuracy` — the percentage of toolcalls whose action JSON was valid
- `goal_completion` — a score derived from the final-answer hit rate (70%, configurable via `agentic.final_completion_weight`) and required-tool usage (30%).
- `error_recovery` — if a toolcall fails and is ran again and then succeeds, it is counted as a recovered error.

### Part 2 (2/3): Roleplay

Each transcript is scored 0–100 by the configured judge(s) against the persona rubric:

`character_consistency`, `instruction_following`, `trap_resistance`, `conversational_quality`.

Responses may be flagged for human review on judge disagreement, boilerplate phrases, **low confidence (<0.55)**, delete-marker flags, heuristic fallback, judge errors, or a seeded random spotcheck.

## Security notes

- The agentic sandbox is a **benchmark harness** and **not a security boundary**. Tasks run locally on your machine, and the model-under-test can invoke allowed shell commands.
This includes `python3` by default, so set `agentic.allow_python3_tool: false` if you don't trust it.

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