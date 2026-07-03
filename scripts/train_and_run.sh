#!/usr/bin/env bash
set -e
python -m src.train --data-path data/raw/sample_churn.csv --model-path models/churn_pipeline.joblib
uvicorn api.main:app --host 0.0.0.0 --port 8000
