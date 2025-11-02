# ==========================================================
# METRICS MODULE - SOUTH GERMAN CREDIT PROJECT
# ==========================================================
from south_german_credit_g57.libraries import *        # Usa tus librerías globales
from south_german_credit_g57.utils.logger import get_logger
from typing import Dict, Tuple


logger = get_logger("MetricsModule")

# ==========================================================
#  FUNCIÓN BASE DE CÁLCULO
# ==========================================================
def calculate_classification_metrics(y_true, y_pred, y_prob=None) -> Dict[str, float]:
    """
    Calcula métricas estándar de clasificación binaria.
    Incluye accuracy, precision, recall, f1, roc_auc y matriz de confusión.
    """
    try:
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
        }

        if y_prob is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
            except ValueError:
                metrics["roc_auc"] = np.nan
                logger.warning(" ROC-AUC no se pudo calcular (probabilidades inválidas).")

        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        return metrics

    except Exception as e:
        logger.error(f" Error al calcular métricas: {e}", exc_info=True)
        return {}

# ==========================================================
# 2️⃣ REGISTRO DE MÉTRICAS EN MLFLOW
# ==========================================================
def log_metrics_mlflow(metrics: Dict[str, float], phase: str = "train"):
    """Registra métricas (excepto la matriz) en MLflow con prefijo por fase."""
    try:
        if not metrics:
            logger.warning(f" No hay métricas para registrar en fase '{phase}'.")
            return

        for key, value in metrics.items():
            if key != "confusion_matrix":
                mlflow.log_metric(f"{phase}_{key}", float(value))

        logger.info(f" Métricas registradas en MLflow para fase '{phase}'.")
    except Exception as e:
        logger.error(f" Error al registrar métricas ({phase}): {e}", exc_info=True)

# ==========================================================
#  MATRIZ DE CONFUSIÓN COMO ARTIFACTO
# ==========================================================
def log_confusion_matrix_plot(y_true, y_pred, phase: str = "train"):
    """Genera y sube la matriz de confusión a MLflow."""
    try:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Matriz de Confusión ({phase})")

        plot_path = f"confusion_matrix_{phase}.png"
        fig.tight_layout()
        fig.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close(fig)

        logger.info(f" Matriz de confusión ({phase}) registrada en MLflow.")
    except Exception as e:
        logger.error(f" Error al generar matriz de confusión ({phase}): {e}", exc_info=True)

# ==========================================================
#  EVALUACIÓN INDIVIDUAL (UNA FASE)
# ==========================================================
def evaluate_and_log(model, X: pd.DataFrame, y: pd.Series, phase: str = "train") -> Dict[str, float]:
    """
    Evalúa un modelo sobre un conjunto específico (train / val / test).
    Calcula métricas, las registra y devuelve el diccionario resultante.
    """
    logger.info(f" Evaluando modelo en fase '{phase}'...")

    try:
        y_pred = model.predict(X)
        y_prob = None
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X)[:, 1]
            except Exception:
                logger.warning(" El modelo no soporta predict_proba correctamente.")

        # Calcular y registrar métricas
        metrics = calculate_classification_metrics(y, y_pred, y_prob)
        log_metrics_mlflow(metrics, phase=phase)
        log_confusion_matrix_plot(y, y_pred, phase=phase)

        logger.info(
            f" [{phase.upper()}] "
            f"Acc: {metrics.get('accuracy',0):.3f} | "
            f"Prec: {metrics.get('precision',0):.3f} | "
            f"Rec: {metrics.get('recall',0):.3f} | "
            f"F1: {metrics.get('f1_score',0):.3f} | "
            f"AUC: {metrics.get('roc_auc',0):.3f}"
        )

        return metrics

    except Exception as e:
        logger.error(f" Error durante la evaluación ({phase}): {e}", exc_info=True)
        return {}

# ==========================================================
#  EVALUACIÓN MULTIFASE AUTOMÁTICA (TRAIN / VAL / TEST)
# ==========================================================
def evaluate_all_phases(model, datasets: Dict[str, Tuple[pd.DataFrame, pd.Series]]) -> Dict[str, Dict[str, float]]:
    """
    Evalúa automáticamente las fases disponibles: train, val, test.
    Si alguna no existe, la omite sin generar errores.
    """
    logger.info(" Iniciando evaluación completa del modelo en fases disponibles...")
    results = {}

    valid_phases = [phase for phase in ["train", "val", "test"] if phase in datasets]
    if not valid_phases:
        logger.warning(" No se proporcionó ningún conjunto de datos para evaluar.")
        return results

    for phase in valid_phases:
        try:
            X, y = datasets[phase]
            results[phase] = evaluate_and_log(model, X, y, phase)
        except Exception as e:
            logger.error(f" Error en fase '{phase}': {e}", exc_info=True)

    logger.info("🏁 Evaluación completada correctamente.")
    return results