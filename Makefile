.PHONY: setup train api test lint docker-build docker-run mlflow clean

setup:
	python -m venv venv
	. venv/Scripts/activate || . venv/bin/activate; pip install -r requirements.txt

train:
	python -m src.train --data-path data/raw/sample_churn.csv --model-path models/churn_pipeline.joblib

api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest -q

lint:
	ruff check api src tests

docker-build:
	docker build -t mlops-churn-api:latest .

docker-run:
	docker run --rm -p 8000:8000 --env-file .env mlops-churn-api:latest

mlflow:
	mlflow ui --host 0.0.0.0 --port 5000

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache api/__pycache__ src/__pycache__ tests/__pycache__
