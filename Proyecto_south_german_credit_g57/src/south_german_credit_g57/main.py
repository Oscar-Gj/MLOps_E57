# ==========================================================
# MAIN PIPELINE ORCHESTRATOR - SOUTH GERMAN CREDIT G57
# ==========================================================
import argparse
import subprocess
import sys
import os
import logging
from datetime import datetime
from pathlib import Path
from importlib import metadata

# --- Configuración del entorno y rutas ---
import sys, os
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# --- Imports del proyecto ---
from south_german_credit_g57.libraries import *
from south_german_credit_g57.preprocessing.clean_data import load_data, clean_data
from south_german_credit_g57.preprocessing.pipeline import build_pipeline
from south_german_credit_g57.utils.dvc_utils import dvc_session
# --- Rutas base ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# --- Imports del proyecto ---
from south_german_credit_g57.preprocessing.clean_data import main as clean_data_main
from south_german_credit_g57.training.train_model_pip import main as train_main
from south_german_credit_g57.evaluation.metrics import main as eval_main

# --- Logger global ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MainPipeline")


# ==========================================================
# VERIFICACIÓN DE DEPENDENCIAS (SOLO 1ª VEZ)
# ==========================================================
def install_package(package_name: str):
    """Instala o actualiza un paquete vía pip (compatible con macOS y Windows)."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])
        logger.info(f"Instalado o actualizado: {package_name}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error al instalar {package_name}: {e}")
        sys.exit(1)


def verify_and_install_requirements(requirements_path: str):
    """Verifica e instala los paquetes requeridos solo si no se ha hecho antes."""
    marker_path = PROJECT_ROOT / ".requirements_verified"

    # Si ya se verificó previamente, saltar
    if marker_path.exists():
        logger.info("Dependencias ya verificadas previamente. Saltando instalación.")
        return

    if not os.path.exists(requirements_path):
        logger.warning(f"No se encontró {requirements_path}. Se omite la verificación.")
        return

    logger.info("Ejecutando verificación inicial de dependencias...")

    with open(requirements_path, "r") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    fixed = 0
    for line in lines:
        # Evitar módulos internos de MLflow
        if "mlflow-tracing" in line.lower():
            logger.info("Omitiendo 'mlflow-tracing' (viene incluido con MLflow 2.x o no aplica).")
            continue

        if "==" in line:
            name, version = line.split("==", 1)
        else:
            name, version = line, None

        try:
            installed_version = metadata.version(name)
            if version and installed_version != version:
                logger.warning(f"{name}: versión instalada {installed_version}, requerida {version}. Corrigiendo...")
                install_package(f"{name}=={version}")
                fixed += 1
            else:
                logger.info(f"{name} {installed_version} OK")
        except metadata.PackageNotFoundError:
            logger.warning(f"{name} no instalado. Instalando...")
            install_package(f"{name}=={version}" if version else name)
            fixed += 1

    # Crear marcador
    with open(marker_path, "w") as f:
        f.write(f"Verificado el {datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.info(f"Dependencias verificadas correctamente. ({fixed} paquetes instalados o actualizados).")


# ==========================================================
# FUNCIÓN PRINCIPAL
# ==========================================================
def run_pipeline(config_path: str, skip_clean=False, skip_train=False, skip_eval=False, full_eval=False):
    """Ejecuta todas las fases del pipeline de forma controlada."""
    logger.info("Inicio de ejecución del pipeline South German Credit G57.")
    start_time = datetime.now()

    try:
        # 0. Verificar dependencias (solo la primera vez)
        req_path = str(PROJECT_ROOT / "requirements.txt")
        verify_and_install_requirements(req_path)

        # Fase 1: Limpieza de datos
        if not skip_clean:
            logger.info("Fase 1: Limpieza de datos.")
            clean_data_main(config_path)
            logger.info("Fase 1 completada correctamente.")
        else:
            logger.info("Fase 1 omitida (--skip-clean).")

        # Fase 2 y 3: Entrenamiento del modelo
        if not skip_train:
            logger.info("Fase 2-3: Entrenamiento de modelos.")
            train_main(config_path)
            logger.info("Entrenamiento finalizado y registrado en MLflow.")
        else:
            logger.info("Fase de entrenamiento omitida (--skip-train).")

        # Fase 4: Evaluación extendida (opcional)
        if full_eval and not skip_eval:
            logger.info("Fase 4: Evaluación extendida del modelo.")
            eval_main(config_path, model_name=None)
            logger.info("Fase 4 completada y resultados registrados en MLflow.")
        else:
            if full_eval:
                logger.warning("Se indicó --full-eval, pero también --skip-eval. No se ejecutará la evaluación.")
            else:
                logger.info("Fase 4 omitida (no se indicó --full-eval).")

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds() / 60
        logger.info(f"Pipeline completo finalizado en {total_time:.2f} minutos.")

    except Exception as e:
        logger.error(f"Error durante la ejecución del pipeline: {e}", exc_info=True)
        sys.exit(1)


# ==========================================================
# ENTRYPOINT
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orquestador del pipeline completo (Clean -> Train -> Eval).")
    parser.add_argument("--config", type=str, default="params.yaml", help="Ruta al archivo YAML de configuración.")
    parser.add_argument("--skip-clean", action="store_true", help="Omitir la fase de limpieza de datos.")
    parser.add_argument("--skip-train", action="store_true", help="Omitir la fase de entrenamiento de modelo.")
    parser.add_argument("--skip-eval", action="store_true", help="Omitir la fase de evaluación final.")
    parser.add_argument("--full-eval", action="store_true", help="Ejecuta la evaluación extendida (fase 4) al final del pipeline.")
    args = parser.parse_args()

    # Ejecuta DVC pull al inicio y push al final alrededor del pipeline completo.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    with dvc_session(
        repo_path=project_root,
        remote="g57_remote_data",
        pull_force=True,
        push_on_success_only=True,
        verbose=True,
    ):
        run_pipeline(
            config_path=args.config,
            skip_clean=args.skip_clean,
            skip_train=args.skip_train,
            skip_eval=args.skip_eval,
            full_eval=args.full_eval,
        )
