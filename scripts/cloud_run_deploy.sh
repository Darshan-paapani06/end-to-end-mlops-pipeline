#!/usr/bin/env bash
set -e

# Example deployment command for Google Cloud Run.
# Replace PROJECT_ID and REGION before running.
PROJECT_ID="your-gcp-project-id"
REGION="asia-south1"
SERVICE="mlops-churn-api"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE}:latest"

gcloud builds submit --tag "$IMAGE"
gcloud run deploy "$SERVICE"   --image "$IMAGE"   --platform managed   --region "$REGION"   --allow-unauthenticated
