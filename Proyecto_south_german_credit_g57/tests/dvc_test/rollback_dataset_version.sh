#!/bin/bash

export PROJECT_DIR="/c/AlgoriML/tec_mon/cursos/TC5044/code/MLOps_E57/MLOps_E57/Proyecto_south_german_credit_g57/"
git checkout HEAD^1 "${PROJECT_DIR}/data/raw/german_credit_original.csv.dvc"
dvc checkout
git commit "${PROJECT_DIR}/data/raw/german_credit_original.csv.dvc" -m "Manteniendo la versión original del set de datos <German South Credit>"

