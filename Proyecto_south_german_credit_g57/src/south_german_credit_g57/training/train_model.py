import pandas as pd
import numpy as np
import yaml
import argparse
import logging
import mlflow
import mlflow.sklearn
import sys
import os
import warnings

from typing import List, Dict, Any, Tuple

# Preprocesamiento
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from category_encoders import BinaryEncoder
from sklearn.preprocessing import PowerTransformer

# Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Pipelines de Desbalanceo
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import NearMiss
from imblearn.combine import SMOTETomek

# Métricas
from sklearn.metrics import make_scorer
from imblearn.metrics import geometric_mean_score

from mlflow.models.signature import infer_signature

# Ignorar warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
# --- Configuración del Logger ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------
# FUNCIONES DE PREPROCESAMIENTO (Basadas en Notebook, Celda [72])
# -----------------------------------------------------------------

def create_preprocessing_pipeline(config: Dict) -> ColumnTransformer:
    """Crea el ColumnTransformer basado en la configuración."""
    logger.info("Creando pipeline de preprocesamiento...")
    
    features_config = config['features']
    num_cols = features_config['numeric']
    cat_cols = features_config['categorical']
    ord_cols = features_config['ordinal']

    # Pipeline Numérico
    numericas_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler()),
        ('power_transform', PowerTransformer(method='yeo-johnson'))
    ])

    # Pipeline Categórico (Nominal)
    nominales_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('binary_encoder', BinaryEncoder(handle_unknown='value', handle_missing='value'))
    ])

    # Pipeline Categórico (Ordinal)
    ordinales_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('binary_encoder', BinaryEncoder(handle_unknown='value', handle_missing='value'))
    ])

    # Ensamblaje del ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('numericas', numericas_pipe, num_cols),
            ('nominales', nominales_pipe, cat_cols),
            ('ordinales', ordinales_pipe, ord_cols)
        ],
        remainder='passthrough'
    )
    return preprocessor

# -----------------------------------------------------------------
# FUNCIONES AUXILIARES (Configuración, Datos, Modelos)
# -----------------------------------------------------------------

def load_config(config_path: str) -> Dict:
    """Carga la configuración desde un archivo YAML."""
    logger.info(f"Cargando configuración desde: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_training_data(path: str, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Carga los datos de entrenamiento (train.csv) y los divide en X, y.
    """
    logger.info(f"Cargando datos de entrenamiento desde: {path}")
    df = pd.read_csv(path)
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    logger.info(f"Datos de entrenamiento cargados. Shape de X: {X.shape}, Shape de y: {y.shape}")
    return X, y

def get_sampler(name: str, params: Dict) -> Any:
    """Instancia el remuestreador basado en el nombre y parámetros."""
    sampler_map = {
        "SMOTE": SMOTE,
        "NearMiss": NearMiss,
        "SMOTETomek": SMOTETomek,
        "SMOTEENN": SMOTEENN
    }
    if name not in sampler_map:
        logger.warning(f"Remuestreador '{name}' no reconocido. Usando 'passthrough'.")
        return 'passthrough'
    
    return sampler_map[name](**params)

def get_model(key: str, params: Dict) -> Any:
    """Instancia el modelo basado en la clave del YAML y sus parámetros."""
    model_map = {
        "logistic_regression": LogisticRegression,
        "knn": KNeighborsClassifier,
        "dtree": DecisionTreeClassifier,
        "rf": RandomForestClassifier,
        "xgboost": XGBClassifier,
        "mlp": MLPClassifier,
        "svc": SVC
    }
    if key not in model_map:
        raise ValueError(f"Clave de modelo '{key}' no reconocida.")
    
    return model_map[key](**params)

def get_scoring_dict(metrics_list: List[str]) -> Dict[str, Any]:
    """Convierte la lista de métricas del YAML en un diccionario para Scikit-Learn."""
    scoring_dict = {}
    simple_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    for metric in metrics_list:
        if metric in simple_metrics:
            scoring_dict[metric.capitalize()] = metric
        elif metric == 'geometric_mean_score':
            scoring_dict['Gmean'] = make_scorer(geometric_mean_score)
        else:
            logger.warning(f"Métrica '{metric}' no reconocida y será ignorada.")
            
    return scoring_dict

# -----------------------------------------------------------------
# FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# -----------------------------------------------------------------

def main(config_path: str):
    """
    Orquestador principal del pipeline de entrenamiento.
    Carga datos, crea pipelines, itera sobre experimentos
    y registra todo en MLflow.
    """
    # 1. Cargar Configuración y Datos de Entrenamiento
    config = load_config(config_path)
    X_train, y_train = load_training_data(
        config['data']['train'],
        config['base']['target_col']
    )
    logger.info(f"Datos de entrenamiento listos: X_train shape {X_train.shape}")

    # 2. Crear Artefactos Reutilizables
    preprocessor = create_preprocessing_pipeline(config)
    
    eval_config = config['evaluation']
    cv = RepeatedStratifiedKFold(
        n_splits=eval_config['cv_splits'],
        n_repeats=eval_config['cv_repeats'],
        random_state=eval_config['cv_random_state']
    )
    
    scoring = get_scoring_dict(eval_config['metrics'])

    # 3. Configurar Experimento MLflow
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    logger.info("Iniciando bucle de experimentos...")

    # 4. Bucle de Experimentos (el núcleo del script)
    for experiment_key, exp_config in config['experiments'].items():
        
        model_name = exp_config['name']
        logger.info(f"--- Ejecutando Experimento: {model_name} ---")
        
        try:
            # Iniciar un "Run" de MLflow
            with mlflow.start_run(run_name=model_name):
                
                # Instanciar modelo y sampler
                model = get_model(experiment_key, exp_config['model_params'])
                sampler = get_sampler(exp_config['resampler'], exp_config['resampler_params'])

                # Registrar Parámetros
                mlflow.log_params(exp_config['model_params'])
                mlflow.log_param("resampler", exp_config['resampler'])
                mlflow.log_param("resampler_params", exp_config['resampler_params'])

                # Crear el pipeline completo (Preprocesador + Sampler + Modelo)
                full_pipeline = ImbPipeline(steps=[
                    ('preprocesador', preprocessor),
                    ('sub_sobre_muestreo', sampler),
                    ('model', model)
                ])

                # --- 1. EVALUACIÓN (para métricas) ---
                logger.info(f"Ejecutando Cross-Validation para {model_name}...")
                scores = cross_validate(
                    full_pipeline,
                    X_train,
                    y_train,
                    scoring=scoring,
                    cv=cv,
                    return_train_score=False # Solo nos importan los scores de test
                )

                # Registrar Métricas (promedio y desviación estándar)
                logger.info(f"Registrando métricas para {model_name}...")
                for metric_name, score_values in scores.items():
                    if "test_" in metric_name:
                        metric_key = metric_name.replace("test_", "")
                        mlflow.log_metric(f"{metric_key}_mean", np.mean(score_values))
                        mlflow.log_metric(f"{metric_key}_std", np.std(score_values))

                # --- 2. ENTRENAMIENTO (para guardar el modelo) --
                logger.info(f"Ajustando el pipeline final en X_train para {model_name}...")
                full_pipeline.fit(X_train, y_train)
                logger.info("Ajuste completado.")

                # --- 3. FIRMA Y REGISTRO DEL MODELO ---
                logger.info("Generando firma del modelo...")
                y_pred_signature = full_pipeline.predict(X_train)
                signature = infer_signature(X_train, y_pred_signature)

                logger.info("Registrando el modelo en MLflow...")
                mlflow.sklearn.log_model(
                    sk_model=full_pipeline,
                    artifact_path="model_pipeline", 
                    signature=signature,
                    input_example=X_train.iloc[:5],
                    registered_model_name=f"{model_name}_model"
                )
                
                logger.info(f"Experimento {model_name} finalizado y registrado.")

        except Exception as e:
            logger.error(f"Error en el experimento {model_name}: {e}")

    logger.info("--- Todos los experimentos han finalizado ---")
    logger.info(f"Ejecuta 'mlflow ui' en tu terminal para ver los resultados.")

# -----------------------------------------------------------------
# PUNTO DE ENTRADA DEL SCRIPT
# -----------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de entrenamiento de modelos para el proyecto de riesgo crediticio.")
    parser.add_argument('--config', type=str, default='params.yaml', help='Ruta al archivo de configuración YAML.')
    
    args = parser.parse_args()
    main(config_path=args.config)