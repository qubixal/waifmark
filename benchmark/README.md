# WAIFMARK© 2 Benchmarking Suite

This directory is the Waifmark 2 benchmark suite. See the [root README](../README.md) for quickstart, scoring, and security notes.

The exact test bank is proprietary and is not committed; run with `data/test_bank.example.json` or provide your own `data/test_bank.json`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run ui/app.py
```