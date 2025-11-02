# ==========================================================
# MAIN PIPELINE EXECUTION - FASE 3 (Entrenamiento con Pipeline)
# ==========================================================

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


def main():
    # Cargar parámetros
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    # Cargar y limpiar datos
    df = load_data(params["data"]["raw"])
    df = clean_data(df, params)

    # Separar features y target
    target_col = params["base"]["target_col"]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Dividir dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["preprocessing"]["test_size"],
        random_state=params["preprocessing"]["random_state"],
        stratify=y,
    )

    # Definir columnas y modelo desde YAML
    numeric_features = params["features"]["numeric"] + params["features"]["ordinal"]
    categorical_features = params["features"].get("categorical", [])
    model_params = params["experiments"]["rf"]["model_params"]  # Random Forest

    # Construir Pipeline
    pipeline = build_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        model_params=model_params,
    )

    # =============================================================
    #  ENTRENAMIENTO, EVALUACIÓN Y REGISTRO EN MLFLOW
    # =============================================================
    with mlflow.start_run(run_name="CreditRisk_RF_Pipeline"):
        logger.info("Entrenando pipeline...")
        pipeline.fit(X_train, y_train)

        logger.info("Evaluando pipeline...")
        y_pred = pipeline.predict(X_test)

        # --- Métricas ---
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("\nRESULTADOS:")
        print(classification_report(y_test, y_pred))

        # --- Log de métricas en el servidor MLflow ---
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # --- Log de parámetros ---
        mlflow.log_params(model_params)

        # --- Guardar modelo solo en local ---
        local_model_path = "models/pipeline_credit.pkl"
        Path("models").mkdir(exist_ok=True)
        joblib.dump(pipeline, local_model_path)
        logger.info(f"Modelo guardado localmente en '{local_model_path}'.")

        logger.info("Métricas registradas en MLflow remoto (sin subir modelo).")


if __name__ == "__main__":
    # Ejecuta DVC pull al inicio y push al final para registrar cambios de dataset
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    with dvc_session(repo_path=project_root, push_on_success_only=True, verbose=True):
        main()
