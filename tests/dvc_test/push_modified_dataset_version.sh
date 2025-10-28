#!/bin/bash

export PROJECT_DIR="/c/AlgoriML/tec_mon/cursos/TC5044/code/MLOps_E57/MLOps_E57/Proyecto_south_german_credit_g57/"
dvc add "${PROJECT_DIR}/data/raw/german_credit_original.csv"
git add "${PROJECT_DIR}/data/raw/german_credit_original.csv.dvc"
git commit -m "Probando las actualizaciones con DVC en el archivo original."
dvc push