# ==========================================================
# METRICS MODULE - SOUTH GERMAN CREDIT PROJECT (Versión Final)
# ==========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
import mlflow
import logging

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from imblearn.metrics import geometric_mean_score

logger = logging.getLogger("MetricsModule")

# ==========================================================
# 1️⃣ FUNCIÓN BASE DE CÁLCULO
# ==========================================================
def calculate_classification_metrics(y_true, y_pred, y_prob=None) -> Dict[str, float]:
    """Calcula métricas estándar de clasificación binaria."""
    try:
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "gmean": geometric_mean_score(y_true, y_pred),
        }

        if y_prob is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
            except ValueError:
                metrics["roc_auc"] = np.nan
                logger.warning("ROC-AUC no se pudo calcular (probabilidades inválidas).")

        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
        return metrics

    except Exception as e:
        logger.error(f"Error al calcular métricas: {e}", exc_info=True)
        return {}

# ==========================================================
# 2️⃣ REGISTRO DE MÉTRICAS EN MLFLOW
# ==========================================================
def log_metrics_mlflow(metrics: Dict[str, float], phase: str = "train"):
    """Registra métricas (excepto la matriz) en MLflow con prefijo por fase."""
    if not metrics:
        logger.warning(f"No hay métricas para registrar en fase '{phase}'.")
        return
    try:
        for key, value in metrics.items():
            if key != "confusion_matrix":
                mlflow.log_metric(f"{phase}_{key}", float(value))
        logger.info(f"Métricas registradas en MLflow para fase '{phase}'.")
    except Exception as e:
        logger.error(f"Error al registrar métricas ({phase}): {e}", exc_info=True)

# ==========================================================
# 3️⃣ MATRIZ DE CONFUSIÓN COMO ARTIFACTO
# ==========================================================
def log_confusion_matrix_plot(y_true, y_pred, phase: str = "train"):
    """Genera y sube la matriz de confusión a MLflow."""
    try:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.set_title(f"Matriz de Confusión ({phase})")
        plot_path = f"confusion_matrix_{phase}.png"
        fig.tight_layout()
        fig.savefig(plot_path)
        mlflow.log_artifact(plot_path, artifact_path="plots")
        plt.close(fig)
        logger.info(f"Matriz de confusión ({phase}) registrada en MLflow.")
    except Exception as e:
        logger.error(f"Error al generar matriz de confusión ({phase}): {e}", exc_info=True)

# ==========================================================
# 4️⃣ EVALUACIÓN INDIVIDUAL (UNA FASE)
# ==========================================================
def evaluate_and_log(model, X: pd.DataFrame, y: pd.Series, phase: str = "train") -> Dict[str, float]:
    """Evalúa un modelo sobre un conjunto específico (train / test / val)."""
    logger.info(f"Evaluando modelo en fase '{phase}'...")
    try:
        y_pred = model.predict(X)
        y_prob = None
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X)[:, 1]
            except Exception:
                logger.warning("El modelo no soporta predict_proba correctamente.")

        metrics = calculate_classification_metrics(y, y_pred, y_prob)
        log_metrics_mlflow(metrics, phase=phase)
        log_confusion_matrix_plot(y, y_pred, phase=phase)

        logger.info(
            f"[{phase.upper()}] "
            f"Acc: {metrics.get('accuracy',0):.3f} | "
            f"Prec: {metrics.get('precision',0):.3f} | "
            f"Rec: {metrics.get('recall',0):.3f} | "
            f"F1: {metrics.get('f1',0):.3f} | "
            f"AUC: {metrics.get('roc_auc',0):.3f}"
        )
        return metrics

    except Exception as e:
        logger.error(f"Error durante la evaluación ({phase}): {e}", exc_info=True)
        return {}

# ==========================================================
# 5️⃣ EVALUACIÓN MULTIFASE AUTOMÁTICA
# ==========================================================
def evaluate_all_phases(model, datasets: Dict[str, Tuple[pd.DataFrame, pd.Series]]) -> Dict[str, Dict[str, float]]:
    """Evalúa automáticamente train, val y test si están presentes."""
    logger.info("Iniciando evaluación multifase del modelo...")
    results = {}
    for phase in ["train", "val", "test"]:
        if phase in datasets:
            try:
                X, y = datasets[phase]
                results[phase] = evaluate_and_log(model, X, y, phase)
            except Exception as e:
                logger.error(f"Error en fase '{phase}': {e}", exc_info=True)
    logger.info("Evaluación multifase completada.")
    return results
