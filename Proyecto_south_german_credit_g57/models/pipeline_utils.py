# ============================================
#  Módulo: pipeline_utils.py
# --------------------------------------------
# Objetivo:
# - Construir pipelines de modelado (una por modelo)
# - Ejecutar validación cruzada y métricas
# - Registrar resultados y modelos en MLflow
# ============================================

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict, cross_validate
from imblearn.pipeline import Pipeline as ImbPipeline

# ============================================================
# Función base: crea pipeline y evalúa
# ============================================================
def train_and_log_model(
    model,
    model_name: str,
    X,
    y,
    preprocessor,
    sampler=None,
    experiment_name: str = "CreditRisk_MLOps_G57",
    n_splits: int = 5,
    n_repeats: int = 2,
    random_state: int = 57
):
    """
    Entrena, evalúa y registra un modelo en MLflow.
    """

    # Configurar experimento
    mlflow.set_experiment(experiment_name)

    # Construir pipeline
    steps = [("pre", preprocessor)]
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("model", model))
    pipe = ImbPipeline(steps=steps)

    # Validación cruzada
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    with mlflow.start_run(run_name=model_name):
        # ---- Fit + evaluación ----
        scores = cross_validate(pipe, X, y, scoring=scoring, cv=cv, n_jobs=-1, return_train_score=True)

        metrics_mean = {k: np.mean(v) for k, v in scores.items() if k.startswith("test_")}
        metrics_mean = {k.replace("test_", ""): v for k, v in metrics_mean.items()}

        # ---- Log de parámetros ----
        mlflow.log_params({
            "model_class": model.__class__.__name__,
            "sampler": sampler.__class__.__name__ if sampler else "None",
            "cv_splits": n_splits,
            "cv_repeats": n_repeats,
            "random_state": random_state
        })

        # ---- Log de métricas ----
        mlflow.log_metrics(metrics_mean)

        # ---- Reentrenar para loguear modelo final ----
        pipe.fit(X, y)

        # Matriz de confusión OOF
        y_pred = cross_val_predict(pipe, X, y, cv=5)
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        ConfusionMatrixDisplay(cm).plot(ax=ax, cmap="cividis", colorbar=False)
        plt.title(f"Matriz de Confusión — {model_name}")
        mlflow.log_figure(fig, f"{model_name}_cm.png")

        # Guardar modelo completo (pipeline)
        mlflow.sklearn.log_model(pipe, artifact_path="model", registered_model_name=model_name)

        print(f"\n{model_name} registrado en MLflow con métricas:")
        for k, v in metrics_mean.items():
            print(f"   {k}: {v:.4f}")

        return metrics_mean
