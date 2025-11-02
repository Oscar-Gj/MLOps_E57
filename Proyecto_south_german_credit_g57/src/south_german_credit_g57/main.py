# ==========================================================
# MAIN PIPELINE EXECUTION - SOUTH GERMAN CREDIT PROJECT
# ==========================================================
import sys
import os
import argparse
from south_german_credit_g57.utils.logger import get_logger

# ----------------------------------------------------------
# Asegurar acceso al paquete
# ----------------------------------------------------------
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# ----------------------------------------------------------
#  Importar fases del pipeline
# ----------------------------------------------------------
from south_german_credit_g57.preprocessing.clean_data import main as clean_main
from south_german_credit_g57.training.train_model_pip import main as train_main
from south_german_credit_g57.evaluation.metrics import evaluate_all_phases

# ----------------------------------------------------------
# Logger general del pipeline
# ----------------------------------------------------------
logger = get_logger("MainPipeline")

# ==========================================================
# FUNCIÓN PRINCIPAL
# ==========================================================
def main(config_path: str, skip_clean: bool = False, skip_train: bool = False):
    """
    Orquestador principal del proyecto South German Credit.

    Parámetros:
    - config_path: Ruta al archivo params.yaml
    - skip_clean: Si True, omite la limpieza de datos.
    - skip_train: Si True, omite el entrenamiento de modelos.
    """
    logger.info("=== INICIO DEL PIPELINE COMPLETO ===")

    #  Etapa de limpieza
    if not skip_clean:
        logger.info("--- Etapa 1: Limpieza y validación de datos ---")
        try:
            clean_main(config_path)
            logger.info("✔ Limpieza completada correctamente.")
        except Exception as e:
            logger.error(f"Error en la limpieza de datos: {e}")
            raise

    # 2️⃣ Etapa de entrenamiento
    if not skip_train:
        logger.info("--- Etapa 2: Entrenamiento y optimización del modelo ---")
        try:
            train_main(config_path)
            logger.info("✔ Entrenamiento completado correctamente.")
        except Exception as e:
            logger.error(f"Error en el entrenamiento: {e}")
            raise

    #  Evaluación general (si aplica)
    try:
        logger.info("--- Etapa 3: Evaluación de métricas finales ---")
        evaluate_all_phases(config_path)
        logger.info("✔ Evaluación completada correctamente.")
    except Exception as e:
        logger.warning(f"⚠ No se pudo ejecutar la evaluación general: {e}")

    logger.info("=== PIPELINE COMPLETO FINALIZADO ===")

# ==========================================================
# PUNTO DE ENTRADA
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline completo del proyecto South German Credit.")
    parser.add_argument("--config", type=str, default="params.yaml", help="Ruta al archivo YAML de configuración.")
    parser.add_argument("--skip_clean", action="store_true", help="Omitir la etapa de limpieza de datos.")
    parser.add_argument("--skip_train", action="store_true", help="Omitir la etapa de entrenamiento de modelos.")
    args = parser.parse_args()

    main(config_path=args.config, skip_clean=args.skip_clean, skip_train=args.skip_train)
