# WAIFMARK© 2 Benchmarking Suite

This directory is the Waifmark 2 benchmark suite. See the [root README](../README.md) for quickstart, scoring, and security notes.

The exact test bank is proprietary and is not committed; run with `data/test_bank.example.json` or provide your own `data/test_bank.json`.

```bash
# One-command boot (new default)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add OPENROUTER_API_KEY / HF_TOKEN if you have them
waifmark              # or: python -m api.run
# → http://127.0.0.1:8001  +  http://127.0.0.1:8001/docs  +  http://127.0.0.1:8001/chart
```

Headless: `python main.py --config config.yaml --test-bank data/test_bank.json`

Judge v2: calibrated prompt with anchor 0-100, confidence-weighted aggregation, heuristic fallback, low-confidence triage (<0.55).
