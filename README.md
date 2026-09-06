<div align="center">

# WAIFMARK 2 <small>Benchmarking Suite</small>

![Banner](/readme_media/banner.png)

![PyPI Python Version](https://img.shields.io/pypi/pyversions/waifmark)

A benchmark that tests the **local agentic** and **roleplay (persona)** capabilities of small (V)LLMs.
</div>

As Waifmark benchmarks _small, locally-hosted models_, cost is not a major concern. Instead, Waifmark 2's scores are evaluated from 0.00 to 100.00 against time per response (total toks ouput / avg. tok/s).

Waifmark has now been updated to **v2**! A lot of things have been changed from **v1**, and so their scores are not directly comparable (see Leaderboard).

## Why v2?

- 4x the question bank compared to v1
- Full automated benchmarking process (that launches vllm/llama.cpp server, serves the local model and benchmarks it in the same control center)
- Ability to test models consecutively, saving setup time
- LLM judge that auto-scores tasks, saving review time. Low confidence evals will still be flagged for a human to audit.

v2 consists of two main types of questions:
- **Agentic** (i.e. short-context toolcalling, strict formatted JSON output, lookup tables, safe shell cmds, code correction)
- **Persona** (i.e. multi-turn conversational context, personality persistance, safety, 中/EN vocabulary and expression)

> Note: Waifmark's exact test bank is **proprietary** and is therefore not released for public. 

## Install & Boot

### 1: Recommended

Download the latest `waifmark-macos.zip` from [Releases](https://github.com/ldpleo/waifmark/releases), unzip, and run:

```bash
./waifmark/waifmark              # launches frontend at http://127.0.0.1:8001
./waifmark/waifmark --stop       # shut down
```

### 2: pip install (Python 3.10+ required)

```bash
pip install waifmark
waifmark                         # launches frontend at http://127.0.0.1:8001
waifmark --stop                  # shut down
```

### 3: From source

```bash
cd benchmark
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # install reqs
waifmark                               # ← boots Frontend app at http://127.0.0.1:8001
```

To shut down:
```bash
waifmark --stop
```

### Env for API keys

```bash
cp .env.example .env
# then, nano into the file or open a code editor to enter your
# OPENROUTER_API_KEY=sk-or-…, HF_TOKEN=hf_…
```

### Headless CLI (No frontend)

If for example you would like to run consecutive benchmarks, or don't want to use the frontend UI:

```bash
python main.py --config config.yaml --test-bank data/test_bank.json
```

Then use `--mode pipeline` to start/stop the backend (vLLM, llama.cpp) that's serving the model.

### Scoring Judge

To speed up the scoring process, you would need to set a judge.

By default, this is done through Openrouter, which offers generous free limits:
1. get an api key from [Openrouter](https://openrouter.ai)
2. set `model` in `config.yaml` under `judges` to the judge model you want (e.g. `openrouter/free`)
3. point `model_under_test` at your model or GGUF file.

## Scoring (v2)

### Agentic (1/3 of test bank)

Waifmark first puts the model in a sandbox to perform agentic tasks, recording and checking each step's tool call.
There are three components that make up the final score:

```
score_100 = 100 × ( tool_syntax_accuracy × 0.3
                  + goal_completion        × 0.5
                  + error_recovery         × 0.2 )
```

- `tool_syntax_accuracy` — the percentage of toolcalls whose JSON was valid
- `goal_completion` — a score derived from the final-answer hit rate (70%) and required-tool usage (30%).
- `error_recovery` — a corrected toolcall after the previous one failed.

> Note: Small models are sometimes sensitive to chat templates. If your benchmark scores look unusually bad, check your vLLM `--chat-template` and `config.yaml` `chat_template`.

### Roleplay (2/3 of test bank)

Then, the model is evaluated based on 4 components:

`character_consistency`, `instruction_following`, `trap_resistance`, `conversational_quality`.

To determine the final Roleplay score. Responses may be flagged for human review on judge disagreement, **low confidence (<0.55)**, judge errors, etc.

## Security Note

- The agentic sandbox is a **benchmark harness** and **not a security boundary**. Tasks run locally on your machine, and the model-under-test can invoke allowed shell commands.
This includes `python3` by default, so set `agentic.allow_python3_tool: false` if you don't trust it.

## Project layout (summarised with AI):

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
  make_dist.py   build wheel + executable
  waifmark.spec  PyInstaller config for standalone build
```