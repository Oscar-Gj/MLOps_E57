# ==========================================================
# TRAIN MODEL PIPELINE - SOUTH GERMAN CREDIT (Versión Híbrida)
# ==========================================================
from south_german_credit_g57.libraries import *
from south_german_credit_g57.utils.logger import get_logger
from south_german_credit_g57.evaluation.metrics_module import calculate_classification_metrics

from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.metrics import geometric_mean_score
from category_encoders import BinaryEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from typing import Dict, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
import yaml
import os
import chardet
import json
import shutil, stat

def on_rm_error(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    os.remove(path)

logger = get_logger("TrainModel")
warnings.filterwarnings("ignore")

# ==========================================================
# RUTAS DINÁMICAS
# ==========================================================
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
DATA_DIR = PROJECT_ROOT / "data"

def resolve_path(relative_path: str) -> Path:
    path = Path(relative_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path

# ==========================================================
# CLASE AUXILIAR
# ==========================================================
class BinaryEncoderWrapper(BaseEstimator, TransformerMixin):
    """Wrapper seguro para BinaryEncoder (convierte todo a string antes de codificar)."""
    def __init__(self, cols=None):
        self.cols = cols
        self.encoder = BinaryEncoder(cols=self.cols, return_df=True)

    def fit(self, X, y=None):
        X_ = X.copy().astype(str)
        self.encoder.fit(X_, y)
        return self

    def transform(self, X):
        X_ = X.copy().astype(str)
        return self.encoder.transform(X_)

# ==========================================================
# FUNCIONES BASE
# ==========================================================
def load_config(path: str) -> Dict:
    """Carga YAML asegurando UTF-8."""
    p = Path(path)
    raw = p.read_bytes()
    enc_info = chardet.detect(raw)
    detected_enc = enc_info.get("encoding", "utf-8")
    if detected_enc.lower() != "utf-8":
        text = raw.decode(detected_enc, errors="ignore")
        p.write_text(text, encoding="utf-8")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_data(path: str, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Carga dataset desde CSV o Parquet."""
    full_path = resolve_path(path)
    logger.info(f"Leyendo dataset desde: {full_path}")
    if not full_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {full_path}")
    if full_path.suffix == ".parquet":
        df = pd.read_parquet(full_path)
    else:
        df = pd.read_csv(full_path)
    return df.drop(columns=[target_col]), df[target_col]

def create_preprocessor(config: Dict) -> ColumnTransformer:
    """Crea pipeline de preprocesamiento."""
    cfg = config["preprocessing"]
    logger.info("Creando pipeline de preprocesamiento...")

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=cfg["numeric"]["imputer_strategy"])),
        ("scaler", MinMaxScaler()),
        ("power", PowerTransformer(method="yeo-johnson"))
    ])
    nom_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=cfg["nominal"]["imputer_strategy"])),
        ("encoder", BinaryEncoderWrapper())
    ])
    ord_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=cfg["ordinal"]["imputer_strategy"])),
        ("scaler", MinMaxScaler())
    ])

    return ColumnTransformer([
        ("num", num_pipe, cfg["numeric"]["features"]),
        ("nom", nom_pipe, cfg["nominal"]["features"]),
        ("ord", ord_pipe, cfg["ordinal"]["features"])
    ], remainder="drop")

def get_model_class(name: str):
    models = {
        "LogisticRegression": LogisticRegression,
        "KNeighborsClassifier": KNeighborsClassifier,
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "RandomForest": RandomForestClassifier,
        "XGBoost": XGBClassifier,
        "MLP": MLPClassifier,
        "SVC": SVC
    }
    return models[name]

def get_sampler_class(name: str):
    samplers = {"SMOTE": SMOTE, "SMOTEENN": SMOTEENN, "NearMiss": NearMiss, "SMOTETomek": SMOTETomek}
    return samplers.get(name, None)

def get_scoring():
    gmean_scorer = make_scorer(geometric_mean_score, greater_is_better=True, average="binary")
    return {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score),
        "recall": make_scorer(recall_score),
        "f1": make_scorer(f1_score),
        "roc_auc": "roc_auc",
        "gmean": gmean_scorer
    }

# ==========================================================
# FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# ==========================================================
def main(config_path: str):
    config = load_config(config_path)
    X, y = load_data(config["data"]["train"], config["base"]["target_col"])
    preprocessor = create_preprocessor(config)

    gs_cfg = config["grid_search"]
    cv = RepeatedStratifiedKFold(
        n_splits=gs_cfg["cv"],
        n_repeats=gs_cfg["n_repeats"],
        random_state=config["base"]["random_state"]
    )

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    logger.info("Conectado a MLflow.")

    for key, model_cfg in config["training"].items():
        model_name = model_cfg["name"]
        logger.info(f"Entrenando modelo: {model_name}")

        try:
            with mlflow.start_run(run_name=model_name, nested=True):
                # Pipeline completo
                model = get_model_class(model_name)()
                sampler_cls = get_sampler_class(model_cfg["resampler"])
                steps = [("preprocessor", preprocessor)]
                if sampler_cls:
                    steps.append(("sampler", sampler_cls(**model_cfg.get("resampler_params", {}))))
                steps.append(("model", model))
                pipeline = ImbPipeline(steps=steps)

                # GridSearchCV
                grid = GridSearchCV(
                    estimator=pipeline,
                    param_grid=model_cfg["param_grid"],
                    scoring=gs_cfg["scoring"],
                    cv=cv,
                    n_jobs=gs_cfg["n_jobs"],
                    verbose=gs_cfg["verbose"]
                )
                grid.fit(X, y)

                best_model = grid.best_estimator_
                best_score = grid.best_score_
                mlflow.log_params(grid.best_params_)
                mlflow.log_metric("best_cv_score", best_score)
                logger.info(f"Mejor score: {best_score:.4f}")

                # Evaluación cruzada completa
                scores = cross_validate(best_model, X, y, cv=cv, scoring=get_scoring())
                metrics_avg = {f"cv_{k.replace('test_','avg_')}": np.mean(v) for k, v in scores.items() if "test_" in k}
                mlflow.log_metrics(metrics_avg)

                # Métricas personalizadas
                y_pred = best_model.predict(X)
                y_prob = best_model.predict_proba(X)[:, 1] if hasattr(best_model, "predict_proba") else None
                custom_metrics = calculate_classification_metrics(y, y_pred, y_prob)
                mlflow.log_metrics({f"train_{k}": v for k, v in custom_metrics.items() if k != "confusion_matrix"})

                # Registro detallado del modelo con fallback
                try:
                    signature = infer_signature(X, best_model.predict(X))
                    mlflow.sklearn.log_model(
                        best_model,
                        artifact_path="model_pipeline",
                        signature=signature,
                        input_example=X.iloc[:5],
                        registered_model_name=f"{model_name}_model"
                    )
                    logger.info(f"Modelo {model_name} registrado correctamente en MLflow.")
                except Exception as e_log:
                    logger.error(f"Error al registrar el modelo en MLflow: {e_log}")
                    fallback_dir = PROJECT_ROOT / "models_fallback"
                    fallback_dir.mkdir(exist_ok=True)
                    local_path = fallback_dir / f"{model_name}_fallback"
                    if local_path.exists():
                        shutil.rmtree(local_path, onerror=on_rm_error)
                    mlflow.sklearn.save_model(best_model, str(local_path))


                    mlflow.log_param("local_model_path", str(local_path))
                    logger.info(f"Modelo guardado localmente en {local_path}")

        except Exception as e:
            logger.error(f"Error en el entrenamiento del modelo {model_name}: {e}", exc_info=True)
            mlflow.end_run(status="FAILED")

    logger.info("Entrenamiento completado para todos los modelos.")

def on_rm_error(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    os.remove(path)