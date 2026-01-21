REPRODUCIBILITY and HOWTO

This file contains the minimal steps to reproduce model training, evaluation and key artifacts for the
AI-First Preauth Fraud Detection project.

Goal
----
Provide a single, simple workflow to reproduce the main artifacts (preprocessing, training, evaluation, and explainability outputs) produced in `artifacts/`.

Prerequisites
-------------
- macOS or Linux
- Python 3.8+ (3.8, 3.9, 3.10 are known to work)
- git

Quick steps (recommended)
-------------------------
1. Create and activate a virtual environment (zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. Run the reproducibility script (it will print the recommended commands and optionally run them):

```bash
python3 scripts/reproduce.py --setup-env --install --dry-run
```

3. To run preprocessing, training, and evaluation (example - adapt to your machine):

```bash
# Preprocess (if a preprocessing script exists)
python3 scripts/generate_incoming_payloads.py || echo "Run your preprocessing step here"

# Train (example - replace with actual train script if present)
python3 scripts/train_xgb.py || echo "Run training script or notebook"

# Evaluate - produce the metrics in artifacts/
python3 scripts/evaluate_models.py || echo "Run evaluation notebook or script"
```

Notes
-----
- This repository includes `requirements.txt` and `requirements.lock.txt`. Prefer using `requirements.lock.txt` for exact reproducibility.
- If any dataset is large or kept externally, follow the pointers in `docs/` to obtain or generate synthetic data.
- For an exact reproduction of artifacts in `artifacts/`, consult the related notebooks in `notebooks/` and the scripts in `scripts/`.

If you have CI available, the minimal checks to include are:
- Install dependencies
- Run a small preprocessing step on a canned (small) sample
- Run a short smoke training (few iterations) to ensure interfaces work
- Run a small inference against `api/main.py` or `dashboard/app.py`

Contact
-------
If anything here fails for you, open an issue or contact the author with the exact failure and environment details.
