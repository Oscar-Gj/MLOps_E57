# ==========================================================
# EVALUACIÓN EXTENDIDA (FASE 4) - SOUTH GERMAN CREDIT PROJECT
# ==========================================================
import argparse
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import sys, os, logging, yaml, chardet
from pathlib import Path

# --- Importar funciones del módulo de métricas ---
from south_german_credit_g57.evaluation.metrics_module import (
    calculate_classification_metrics,
    log_confusion_matrix_plot
)

# --- Configuración de logger global ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("EvaluateModel")


# ==========================================================
# FUNCIONES AUXILIARES
# ==========================================================
def load_config(config_path: str):
    """Carga el archivo YAML asegurando UTF-8."""
    path = Path(config_path)
    raw = path.read_bytes()
    enc_info = chardet.detect(raw)
    detected = enc_info.get("encoding", "utf-8")
    if detected.lower() != "utf-8":
        text = raw.decode(detected, errors="ignore")
        path.write_text(text, encoding="utf-8")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_test_data(path: str, target_col: str):
    """Carga el dataset de prueba desde CSV."""
    df = pd.read_csv(path)
    X_test = df.drop(columns=[target_col])
    y_test = df[target_col]
    logger.info(f"Datos de prueba cargados correctamente. Shape: {df.shape}")
    return X_test, y_test


def find_best_model(config: dict) -> str:
    """Busca en MLflow el mejor modelo del experimento de entrenamiento."""
    mlflow_cfg = config.get("mlflow", {})
    exp_name = mlflow_cfg.get("experiment_name", "Experimento-Conexión-MLFlow-Grupo57")
    metric_name = config["grid_search"]["scoring"]
    sort_metric = f"metrics.avg_cv_{metric_name}"

    logger.info(f"Buscando mejor modelo del experimento '{exp_name}' ordenado por '{sort_metric}'...")

    runs = mlflow.search_runs(
        experiment_names=[exp_name],
        order_by=[f"{sort_metric} DESC"],
        max_results=1
    )
    if runs.empty:
        logger.error("No se encontraron runs válidos en MLflow.")
        sys.exit(1)

    best_run = runs.iloc[0]
    base_model = best_run["tags.mlflow.runName"].replace("_GridSearch", "")
    best_model = f"{base_model}_model"

    logger.info(f"Modelo campeón encontrado: {best_model}")
    return best_model


# ==========================================================
# FUNCIÓN PRINCIPAL
# ==========================================================
def main(config_path: str, model_name: str = None):
    """Evalúa el modelo campeón en el conjunto de prueba y registra resultados en MLflow."""
    # 1️⃣ Cargar configuración
    config = load_config(config_path)
    mlflow_cfg = config.get("mlflow", {})

    # 2️⃣ Configurar MLflow (local o remoto)
    if mlflow_cfg.get("mode", "cloud") == "local":
        uri = "file://" + str(Path("mlruns").resolve())
        mlflow.set_tracking_uri(uri)
        logger.info(f"MLflow configurado en modo LOCAL → {uri}")
    else:
        mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
        logger.info(f"MLflow configurado en modo CLOUD → {mlflow_cfg['tracking_uri']}")

    eval_exp = mlflow_cfg.get("evaluation_experiment_name", "MLFlow-Grupo57-Evaluación")
    mlflow.set_experiment(eval_exp)

    # 3️⃣ Cargar datos de prueba
    X_test, y_test = load_test_data(config["data"]["test"], config["base"]["target_col"])

    # 4️⃣ Determinar modelo a evaluar
    if model_name is None:
        model_name = find_best_model(config)
    model_uri = f"models:/{model_name}/latest"
    logger.info(f"Evaluando modelo: {model_name} ({model_uri})")

    # 5️⃣ Iniciar run de evaluación
    mlflow.end_run()
    with mlflow.start_run(run_name=f"evaluation_{model_name}") as run:
        mlflow.log_param("evaluated_model", model_name)
        mlflow.log_param("evaluation_phase", "final_test")

        try:
            # 6️⃣ Cargar modelo desde MLflow
            model = mlflow.sklearn.load_model(model_uri)
            logger.info("Modelo cargado correctamente desde el registry.")

            # 7️⃣ Generar predicciones
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

            # 8️⃣ Calcular métricas
            metrics = calculate_classification_metrics(y_test, y_pred, y_prob)
            metrics_prefixed = {f"final_test_{k}": v for k, v in metrics.items() if k != "confusion_matrix"}
            mlflow.log_metrics(metrics_prefixed)
            logger.info("Métricas finales registradas en MLflow correctamente.")

            # 9️⃣ Matriz de confusión
            log_confusion_matrix_plot(y_test, y_pred, phase="final_test")

            # 🔟 Guardar reporte local y subir como artefacto
            reports_dir = Path("reports/evaluation_final")
            reports_dir.mkdir(parents=True, exist_ok=True)
            report_path = reports_dir / f"report_{model_name}.txt"

            with open(report_path, "w") as f:
                for k, v in metrics.items():
                    f.write(f"{k}: {v}\n")

            mlflow.log_artifact(str(report_path))
            logger.info(f"Reporte de clasificación guardado y registrado: {report_path}")

        except Exception as e:
            logger.error(f"Error durante la evaluación: {e}", exc_info=True)
            mlflow.end_run(status="FAILED")
            sys.exit(1)

    logger.info("✅ Evaluación finalizada correctamente y registrada en MLflow.")


# ==========================================================
# ENTRYPOINT
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluación extendida de modelo (fase 4).")
    parser.add_argument("--config", type=str, default="params.yaml", help="Ruta al archivo YAML de configuración.")
    parser.add_argument("--model_name", type=str, default=None, help="(Opcional) Nombre del modelo en MLflow.")
    args = parser.parse_args()
    main(config_path=args.config, model_name=args.model_name)
