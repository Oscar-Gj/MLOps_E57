# ==========================================================
# MAIN PIPELINE EXECUTION - FASE 3 (Entrenamiento + Optimización)
# ==========================================================

# --- Configuración del entorno y rutas ---
import sys, os
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)
print(f"Carpeta SRC agregada al PYTHONPATH: {src_path}")

# --- Imports del proyecto ---
from south_german_credit_g57.libraries import *      
from south_german_credit_g57.seed import set_seed, get_random_state
from south_german_credit_g57.train_model import main as train_main

# =============================================================
# Configuración de conexión a servidor MLflow remoto
# =============================================================
MLFLOW_TRACKING_URI = "https://mlflow-super-g57-137680020436.us-central1.run.app"
EXPERIMENT_NAME = "Experimento-Conexión-MLFlow-Grupo57"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# --- Configurar logger ---
logger = get_logger("MainPipeline")

# =============================================================
# FUNCIÓN PRINCIPAL
# =============================================================
def main():
    logger.info("Iniciando ejecución principal del proyecto South German Credit — Fase 3")

    # Fijar semilla global para reproducibilidad
    set_seed()
    RANDOM_STATE = get_random_state()
    logger.info(f"Semilla global fijada en {RANDOM_STATE}")

    # Validar existencia del archivo params.yaml
    CONFIG_PATH = "params.yaml"
    if not os.path.exists(CONFIG_PATH):
        logger.error(f"No se encontró el archivo de configuración en: {CONFIG_PATH}")
        return
    logger.info(f"Archivo de configuración detectado: {CONFIG_PATH}")

    # Cargar parámetros del YAML
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    #  Conexión MLflow (si existe en el YAML)
    if "mlflow" in params:
        mlflow.set_tracking_uri(params["mlflow"].get("tracking_uri", MLFLOW_TRACKING_URI))
        mlflow.set_experiment(params["mlflow"].get("experiment_name", EXPERIMENT_NAME))
        logger.info("🔗 Conectado a servidor MLflow remoto.")
    else:
        logger.warning("No se encontró configuración MLflow en params.yaml. Usando valores por defecto.")

    # Ejecución del pipeline de entrenamiento completo
    try:
        logger.info("Ejecutando pipeline de entrenamiento (GridSearch + MLflow)...")
        train_main(config_path=CONFIG_PATH)
        logger.info("Entrenamiento completado exitosamente.")
    except Exception as e:
        logger.exception("Error durante la ejecución del pipeline principal.")
        return

    # Cierre del flujo
    logger.info("Proceso finalizado correctamente.")
    logger.info("Revisa los resultados en MLflow UI o ejecuta: mlflow ui")

# =============================================================
# EJECUCIÓN
# =============================================================
if __name__ == "__main__":
    main()
