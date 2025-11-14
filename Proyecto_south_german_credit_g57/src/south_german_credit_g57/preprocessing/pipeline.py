# ==========================================================
# PIPELINE DE PREPROCESAMIENTO Y MODELADO - VERSIÓN EXTENDIDA
# ==========================================================
# Compatible con múltiples modelos configurables desde params.yaml.
# Permite añadir nuevos sin modificar el código.
# ==========================================================

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler, OrdinalEncoder, OneHotEncoder, FunctionTransformer
)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import mlflow

# Modelos disponibles
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from south_german_credit_g57.utils.logger import get_logger

logger = get_logger("BuildPipeline")


# ==========================================================
# Función auxiliar: conversión segura a numérico
# ==========================================================
def safe_to_numeric(X):
    """Convierte texto a número, reemplazando errores por NaN."""
    return X.apply(pd.to_numeric, errors="coerce")


# ==========================================================
# Validación de columnas
# ==========================================================
def validate_features(df, features):
    missing = [f for f in features if f not in df.columns]
    if missing:
        logger.warning(f"Columnas faltantes en el dataset: {missing}")


# ==========================================================
# Catálogo de modelos disponibles
# ==========================================================
def get_supported_models():
    """Devuelve el diccionario con los modelos disponibles."""
    return {
        "RandomForest": RandomForestClassifier,
        "LogisticRegression": LogisticRegression,
        "XGBoost": XGBClassifier,
        "LightGBM": LGBMClassifier,
        "GradientBoosting": GradientBoostingClassifier,
        "AdaBoost": AdaBoostClassifier,
        "DecisionTree": DecisionTreeClassifier,
        "SVM": SVC,
        "LinearSVM": LinearSVC,
        "KNN": KNeighborsClassifier
    }


# ==========================================================
# Función principal: construir el pipeline
# ==========================================================
def build_pipeline(
    numeric_features,
    nominal_features,
    ordinal_features,
    model_params
):
    """
    Construye el pipeline de preprocesamiento y modelado configurable.

    Parameters
    ----------
    numeric_features : list
        Columnas numéricas.
    nominal_features : list
        Columnas categóricas nominales.
    ordinal_features : list
        Columnas categóricas ordinales.
    model_params : dict
        Diccionario con hiperparámetros y el nombre del modelo (clave 'name').
    """

    logger.info("Iniciando construcción del pipeline...")

    # ======================================================
    # 1️⃣ Transformadores de columnas
    # ======================================================
    numeric_transformer = Pipeline(steps=[
        ("to_numeric", FunctionTransformer(safe_to_numeric)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    ordinal_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])

    nominal_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # ======================================================
    # 2️⃣ ColumnTransformer
    # ======================================================
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("ord", ordinal_transformer, ordinal_features),
            ("nom", nominal_transformer, nominal_features)
        ],
        remainder="drop"
    )

    # ======================================================
    # 3️⃣ Modelo configurable
    # ======================================================
    models = get_supported_models()
    model_name = model_params.get("name", "RandomForest")

    if model_name not in models:
        raise ValueError(
            f"Modelo '{model_name}' no soportado. "
            f"Modelos disponibles: {list(models.keys())}"
        )

    # Copia de los parámetros sin la clave 'name'
    model_args = {k: v for k, v in model_params.items() if k != "name"}

    model_class = models[model_name]
    model = model_class(**model_args)

    logger.info(f"Modelo seleccionado: {model_name} con parámetros: {model_args}")

    # ======================================================
    # 4️⃣ Pipeline completo
    # ======================================================
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    # ======================================================
    # 5️⃣ Registro en MLflow
    # ======================================================
    try:
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("num_features", len(numeric_features))
        mlflow.log_param("ord_features", len(ordinal_features))
        mlflow.log_param("nom_features", len(nominal_features))
        logger.info("Parámetros del pipeline registrados en MLflow.")
    except Exception as e:
        logger.warning(f"No se pudo registrar el pipeline en MLflow: {e}")

    logger.info("✅ Pipeline construido correctamente.")
    return pipeline
