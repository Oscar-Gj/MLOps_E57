# ==========================================================
# MAIN PIPELINE ORCHESTRATOR - SOUTH GERMAN CREDIT PROJECT
# ==========================================================
import argparse
import sys
import os
import subprocess
import mlflow
import yaml
from datetime import datetime
from importlib import metadata

# ==========================================================
# AJUSTE DINÁMICO DE RUTAS
# ==========================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# === Imports del proyecto ===
from south_german_credit_g57.utils.seed import set_seed, get_random_state
from south_german_credit_g57.utils.logger import get_logger
from south_german_credit_g57.preprocessing.clean_data import main as clean_data_main
from south_german_credit_g57.training.train_model_pip import main as train_main
from south_german_credit_g57.evaluation.metrics import main as eval_main
from south_german_credit_g57.evaluation.metrics_module import evaluate_all_phases

logger = get_logger("MainPipeline")

# ==========================================================
# INSTALACIÓN DE DEPENDENCIAS (moderno sin pkg_resources)
# ==========================================================
def _install_package(package_name: str):
    """Instala o actualiza un paquete vía pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        logger.info(f"Instalado o actualizado: {package_name}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error al instalar {package_name}: {e}")
        sys.exit(1)


def verify_and_install_requirements(requirements_path="requirements.txt"):
    """
    Verifica dependencias e instala/actualiza si es necesario.
    Solo se ejecuta la primera vez; crea un archivo '.requirements_verified' como bandera.
    Además, omite 'mlflow-tracing' porque no debe instalarse manualmente en 2.x.
    """
    marker_path = os.path.join(PROJECT_ROOT, ".requirements_verified")

    # Si ya se verificó previamente, saltar
    if os.path.exists(marker_path):
        logger.info("Dependencias ya verificadas previamente. Saltando verificación.")
        return

    logger.info("Ejecutando verificación inicial de dependencias...")

    if not os.path.exists(requirements_path):
        logger.warning(f"No se encontró {requirements_path}. Se omite la verificación.")
        return

    with open(requirements_path, "r") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    total = len(lines)
    fixed = 0

    for line in lines:
        # Evitar módulos que no deben instalarse manualmente con MLflow 2.x
        if "mlflow-tracing" in line.lower():
            logger.info("Omitiendo 'mlflow-tracing' (viene incluido con mlflow 2.x o no aplica).")
            continue

        if "==" in line:
            name, version = line.split("==", 1)
        else:
            name, version = line, None

        try:
            installed_version = metadata.version(name)
            if version and installed_version != version:
                logger.warning(f"{name}: instalada {installed_version}, requerida {version}. Corrigiendo...")
                _install_package(f"{name}=={version}")
                fixed += 1
            else:
                logger.info(f"{name} {installed_version} OK")
        except metadata.PackageNotFoundError:
            logger.warning(f"{name} no instalado. Instalando...")
            _install_package(f"{name}=={version}" if version else name)
            fixed += 1

    # Crear bandera de verificación completada
    with open(marker_path, "w") as f:
        f.write(f"Verificado el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if fixed > 0:
        logger.info(f"{fixed}/{total} dependencias corregidas. Reinicia el script para aplicar cambios.")
        sys.exit(0)
    else:
        logger.info("Todas las dependencias están correctas.")
        logger.info(f"Marcador creado en: {marker_path}")

# ==========================================================
# VALIDACIÓN AUTOMÁTICA DE ENCODING DE CONFIGURACIÓN
# ==========================================================
from pathlib import Path
import chardet

def ensure_utf8_encoding(file_path: str):
    """Verifica y reescribe el archivo en UTF-8 si se detecta otro encoding."""
    try:
        path = Path(file_path)
        raw = path.read_bytes()
        enc_info = chardet.detect(raw)
        detected = enc_info.get("encoding", "utf-8")

        if detected.lower() != "utf-8":
            logger.warning(f"Archivo {file_path} detectado en {detected}. Corrigiendo a UTF-8...")
            text = raw.decode(detected, errors="ignore")
            path.write_text(text, encoding="utf-8")
            logger.info(f"{file_path} reescrito correctamente en UTF-8.")
        else:
            logger.info(f"{file_path} ya está en UTF-8.")
    except Exception as e:
        logger.warning(f"No se pudo verificar el encoding de {file_path}: {e}")

# ==========================================================
# FUNCIÓN PRINCIPAL DE ORQUESTACIÓN
# ==========================================================
def run_pipeline(config_path: str, skip_clean=False, skip_train=False, skip_eval=False):
    logger.info("Inicio de ejecución del pipeline de South German Credit")

    # 1. Verificar dependencias
    verify_and_install_requirements("requirements.txt")

    # 2. Fijar semilla global
    set_seed()
    # 3. Validar y cargar configuración YAML
    if not os.path.exists(config_path):
        logger.error(f"No se encontró el archivo de configuración: {config_path}")
        sys.exit(1)

    # --- Validar y corregir encoding automáticamente ---
    ensure_utf8_encoding(config_path)

    # --- Cargar archivo YAML limpio ---
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)


    mlflow_mode = config.get("mlflow", {}).get("mode", "local")

    if mlflow_mode == "local":
        local_uri = "file://" + os.path.join(PROJECT_ROOT, "mlruns")
        mlflow.set_tracking_uri(local_uri)
        logger.info(f"MLflow configurado en modo LOCAL → {local_uri}")
    else:
        remote_uri = config["mlflow"]["tracking_uri"]
        mlflow.set_tracking_uri(remote_uri)
        logger.info(f"MLflow configurado en modo CLOUD → {remote_uri}")


    # 5. Ejecutar fases principales
    with mlflow.start_run(run_name=f"Full_Run_{datetime.now():%Y%m%d_%H%M%S}") as run:
        mlflow.log_param("run_start", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        mlflow.log_param("random_state", get_random_state())

        # ----------------------------------------------------------
        # Fase 1: Limpieza de datos
        # ----------------------------------------------------------
        if not skip_clean:
            logger.info("Fase 1: Limpieza de datos")
            clean_data_main(config_path)
            mlflow.log_param("phase_clean_data", "completed")
        else:
            logger.info("Fase de limpieza omitida")

        # ----------------------------------------------------------
        # Fase 2: Entrenamiento
        # ----------------------------------------------------------
        if not skip_train:
            logger.info("Fase 2: Entrenamiento y registro de modelos")
            train_main(config_path)
            mlflow.log_param("phase_train_model", "completed")
        else:
            logger.info("Fase de entrenamiento omitida")

        # ----------------------------------------------------------
        # Fase 3: Evaluación final
        # ----------------------------------------------------------
        if not skip_eval:
            logger.info("Fase 3: Evaluación final en test")
            eval_main(config_path, model_name=None)
            mlflow.log_param("phase_evaluation", "completed")
        else:
            logger.info("Fase de evaluación omitida")

        # ----------------------------------------------------------
        # Fase 4: Evaluación local (opcional)
        # ----------------------------------------------------------
        try:
            logger.info("Fase 4: Evaluación local rápida (train/test split)")
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from joblib import load

            df = pd.read_csv(config["data"]["processed"])
            X = df.drop(columns=[config["base"]["target_col"]])
            y = df[config["base"]["target_col"]]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=get_random_state()
            )

            model_path = "models/random_forest_model.pkl"
            if os.path.exists(model_path):
                model = load(model_path)
                datasets = {"train": (X_train, y_train), "test": (X_test, y_test)}
                results = evaluate_all_phases(model, datasets)
                logger.info(f"Resultados locales: {results}")

                # 🔸 Registrar métricas locales en MLflow
                for phase, metrics_dict in results.items():
                    for metric_name, metric_value in metrics_dict.items():
                        mlflow.log_metric(f"local_{phase}_{metric_name}", metric_value)

                logger.info("Métricas locales registradas en MLflow correctamente.")
            else:
                logger.warning("No se encontró modelo local para evaluación rápida.")
        except Exception as e:
            logger.warning(f"No se pudo ejecutar evaluación local: {e}")

        mlflow.log_param("run_end", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        logger.info("Pipeline completado correctamente.")
        logger.info("Consulta resultados en la interfaz MLflow UI.")


# ==========================================================
# ENTRYPOINT
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orquestador del pipeline completo (Clean → Train → Eval).")
    parser.add_argument("--config", type=str, default="params.yaml", help="Ruta al archivo YAML de configuración.")
    parser.add_argument("--skip-clean", action="store_true", help="Omitir la fase de limpieza.")
    parser.add_argument("--skip-train", action="store_true", help="Omitir la fase de entrenamiento.")
    parser.add_argument("--skip-eval", action="store_true", help="Omitir la fase de evaluación.")
    args = parser.parse_args()

    run_pipeline(
        config_path=args.config,
        skip_clean=args.skip_clean,
        skip_train=args.skip_train,
        skip_eval=args.skip_eval,
    )
