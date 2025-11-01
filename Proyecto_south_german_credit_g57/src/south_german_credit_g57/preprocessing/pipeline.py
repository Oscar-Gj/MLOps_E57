# ==========================================================
# PIPELINE DE PREPROCESAMIENTO Y MODELADO
# ==========================================================
# Este script define la construcción del pipeline principal
# bajo las mejores prácticas de Scikit-Learn, con modularidad
# y compatibilidad total con serialización (joblib/pickle).
# ==========================================================

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from south_german_credit_g57.libraries import get_logger

logger = get_logger("BuildPipeline")


# ==========================================================
# Función auxiliar para convertir texto a numérico
# ==========================================================
def safe_to_numeric(X):
    """
    Convierte valores de texto a numéricos, reemplazando errores por NaN.
    Esta función es externa (no lambda) para permitir serialización del pipeline.
    """
    return X.apply(pd.to_numeric, errors="coerce")


# ==========================================================
# Función principal para construir el pipeline
# ==========================================================
def build_pipeline(numeric_features, categorical_features, model_params):
    """
    Construye el pipeline de preprocesamiento y entrenamiento.

    Parameters
    ----------
    numeric_features : list
        Lista de columnas numéricas a transformar.
    categorical_features : list
        Lista de columnas categóricas a transformar.
    model_params : dict
        Diccionario con hiperparámetros del modelo (Random Forest).
    """

    logger.info("Iniciando construcción del pipeline...")

    # --- Transformadores ---
    numeric_transformer = Pipeline(steps=[
        ('to_numeric', FunctionTransformer(safe_to_numeric)),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # --- Composición del preprocesador ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # descarta columnas no especificadas
    )

    # --- Modelo base ---
    model = RandomForestClassifier(**model_params)

    # --- Pipeline completo ---
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    logger.info("Pipeline construido correctamente.")
    return pipeline
