#!/bin/bash

echo "Inicia el Servidor MLFlow usando PostgreSQL y GCP-Bucket"

export MY_SECRET=$(gcloud secrets versions access latest --secret=POST_GRE_PWD --project=laboratorio1-447417)

mlflow server --backend-store-uri postgresql://postgres:${POST_GRE_PWD}@10.84.112.6/mlflow --host 0.0.0.0 --port 5000
