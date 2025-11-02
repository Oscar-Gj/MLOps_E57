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
import json

from typing import List, Dict, Any, Tuple

# Preprocesamiento
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, GridSearchCV
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
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.metrics import geometric_mean_score

from mlflow.models.signature import infer_signature
from mlflow.data.pandas_dataset import PandasDataset

# Ignorar warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
# --- Configuración del Logger ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------
# CLASES Y FUNCIONES DE AYUDA
# -----------------------------------------------------------------

class BinaryEncoderWrapper(BaseEstimator, TransformerMixin):
    """ Wrapper para BinaryEncoder para manejar la conversión de tipos de datos a 'str' """
    def __init__(self, cols=None):
        self.cols = cols
        self.encoder = BinaryEncoder(cols=self.cols, return_df=True)

    def fit(self, X, y=None):
        X_ = X.copy()
        if self.cols:
            for col in self.cols:
                X_[col] = X_[col].astype(str)
        else:
            X_ = X_.astype(str)
        self.encoder.fit(X_, y)
        return self

    def transform(self, X):
        X_ = X.copy()
        if self.cols:
            for col in self.cols:
                X_[col] = X_[col].astype(str)
        else:
            X_ = X_.astype(str)
        return self.encoder.transform(X_)

# -----------------------------------------------------------------
# FUNCIONES DE PREPROCESAMIENTO (Basadas en Notebook, Celda [72])
# -----------------------------------------------------------------

def create_preprocessing_pipeline(config: Dict) -> ColumnTransformer:
    """Construye el pipeline de preprocesamiento basado en la configuración."""
    logger.info("Construyendo pipeline de preprocesamiento...")

    cfg_prep = config['preprocessing']

    # Pipeline para variables numéricas (con transformación Yeo-Johnson)
    numericas_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=cfg_prep['numeric']['imputer_strategy'])),
        ('scaler', MinMaxScaler()),
        ('power_transform', PowerTransformer(method='yeo-johnson'))
    ])

    # Pipeline Categórico (Nominal)
    nominales_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=cfg_prep['nominal']['imputer_strategy'])),
        ('binary_encoder', BinaryEncoderWrapper(cols=None))
    ])

    # Pipeline Categórico (Ordinal)
    ordinales_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=cfg_prep['ordinal']['imputer_strategy'])),
        ('scaler', MinMaxScaler())
    ])

    # Ensamblaje del ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('numericas', numericas_pipe, cfg_prep['numeric']['features']),
            ('nominales', nominales_pipe, cfg_prep['nominal']['features']),
            ('ordinales', ordinales_pipe, cfg_prep['ordinal']['features'])
        ],
        remainder='passthrough'
    )
    logger.info("Pipeline de preprocesamiento construido exitosamente.")
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
    try:
        df = pd.read_csv(path)
        X_train = df.drop(columns=[target_col])
        y_train = df[target_col]
        logger.info(f"Datos de entrenamiento cargados. Shape de X_train: {X_train.shape}")
        return X_train, y_train
    except FileNotFoundError:
        logger.error(f"Error: No se encontró el archivo de datos en {path}.")
        sys.exit(1)

def get_sampler(sampler_name: str) -> Any:
    """Obtiene la clase del sampler a partir de su nombre."""
    samplers = {
        "SMOTE": SMOTE,
        "SMOTEENN": SMOTEENN,
        "NearMiss": NearMiss,
        "SMOTETomek": SMOTETomek,
        "Passthrough": None 
    }
    sampler_class = samplers.get(sampler_name)
    if sampler_class is None and sampler_name != "Passthrough":
        raise ValueError(f"Sampler '{sampler_name}' no reconocido.")
    return sampler_class


def get_model(model_name: str) -> Any:
    """Obtiene la clase del modelo a partir de su nombre."""
    models = {
        "LogisticRegression": LogisticRegression,
        "KNeighborsClassifier": KNeighborsClassifier,
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "RandomForest": RandomForestClassifier,
        "XGBoost": XGBClassifier,
        "MLP": MLPClassifier,
        "SVC": SVC
    }
    model_class = models.get(model_name)
    if model_class is None:
        raise ValueError(f"Modelo '{model_name}' no reconocido.")
    return model_class

def get_scoring_dict() -> Dict[str, Any]:
    """Define las métricas que se usarán en la validación cruzada."""
    gmean_scorer = make_scorer(geometric_mean_score, greater_is_better=True, average='binary')
    
    return {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, average='binary'),
        'recall': make_scorer(recall_score, average='binary'),
        'f1': make_scorer(f1_score, average='binary'),
        'roc_auc': 'roc_auc',
        'gmean': gmean_scorer
    }

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
        path=config['data']['train'],
        target_col=config['base']['target_col']
    )
    
    # 2. Obtener Pipeline de Preprocesamiento
    preprocessor = create_preprocessing_pipeline(config)
    
    # 3. Obtener Configuración de GridSearch y Estrategia de CV
    gs_config = config['grid_search']
    cv_strategy = RepeatedStratifiedKFold(
        n_splits=gs_config['cv'],
        n_repeats=gs_config['n_repeats'],
        random_state=config['base']['random_state']
    )
    
    # 4. Configurar MLflow
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    logger.info("Iniciando bucle de experimentos...")

    # 5. Iterar sobre los modelos definidos en 'training'
    for model_key, model_config in config['training'].items():
        
        model_name = model_config['name']
        logger.info(f"--- Iniciando experimento para: {model_name} ---")

        try:
            with mlflow.start_run(run_name=f"{model_name}_GridSearch") as run:
                
                # --- 1. CONFIGURACIÓN DEL PIPELINE ---
                
                # Obtener la clase del modelo (sin instanciar con parámetros)
                model_class = get_model(model_name)
                model_instance = model_class() 
                
                # Obtener el sampler
                sampler_class = get_sampler(model_config['resampler'])
                if sampler_class:
                    sampler_instance = sampler_class(**model_config.get('resampler_params', {}))
                    pipeline_steps = [
                        ('preprocessor', preprocessor),
                        ('sampler', sampler_instance),
                        ('model', model_instance)
                    ]
                else:
                    # Caso 'Passthrough' (sin sampler)
                    pipeline_steps = [
                        ('preprocessor', preprocessor),
                        ('model', model_instance)
                    ]
                
                # Pipeline completo (estimador para GridSearchCV)
                full_pipeline = ImbPipeline(steps=pipeline_steps)

                # Obtener el grid de parámetros
                param_grid = model_config['param_grid']

                # --- 2. EJECUCIÓN DE GRIDSEARCHCV ---
                
                logger.info(f"Iniciando GridSearchCV para {model_name}...")
                logger.info(f"Métrica de optimización: {gs_config['scoring']}")
                
                grid_search = GridSearchCV(
                    estimator=full_pipeline,
                    param_grid=param_grid,
                    scoring=gs_config['scoring'],
                    cv=cv_strategy,
                    n_jobs=gs_config['n_jobs'],
                    verbose=gs_config['verbose']
                )
                
                # Ajustar el GridSearch
                grid_search.fit(X_train, y_train)
                
                logger.info(f"GridSearchCV para {model_name} finalizado.")
                
                # --- 3. REGISTRO DE RESULTADOS DE GRIDSEARCH ---
                
                # Obtener el mejor pipeline
                best_pipeline = grid_search.best_estimator_
                best_params = grid_search.best_params_
                best_score = grid_search.best_score_
                
                logger.info(f"Mejor score ({gs_config['scoring']}): {best_score:.4f}")
                logger.info(f"Mejores parámetros encontrados: {best_params}")
                
                # Registrar parámetros de GS y mejores parámetros
                mlflow.log_params(gs_config)
                mlflow.log_param(f"{model_name}_param_grid", json.dumps(param_grid))
                mlflow.log_param(f"{model_name}_best_params", json.dumps(best_params))
                mlflow.log_metric(f"best_cv_{gs_config['scoring']}", best_score)
                # Registrar dataset en MLflow
                try:
                    dataset_d = pd.read_csv(config['data']['train'])
                    dataset_dict = mlflow.data.from_pandas(
                        dataset_d, source=config['data']['train'], name="South German Credit", targets="credit_risk",
                    )
                    mlflow.log_input(dataset_dict, context="training")
                    logger.info("Dataset de entrenamiento registrado como artifact en MLflow.")
                except Exception as e:
                    logger.warning(f"No se pudo registrar el dataset en MLflow: {e}")

                # --- 4. RE-EVALUACIÓN DEL MEJOR MODELO CON MÚLTIPLES MÉTRICAS ---
                logger.info(f"Re-evaluando el mejor estimador con {gs_config['cv']} folds y {gs_config['n_repeats']} repeticiones para obtener todas las métricas...")
                
                scoring_metrics = get_scoring_dict()
                cv_results = cross_validate(
                    best_pipeline, # Usamos el mejor pipeline encontrado
                    X_train,
                    y_train,
                    cv=cv_strategy, # Usamos la misma estrategia de CV
                    scoring=scoring_metrics,
                    n_jobs=gs_config['n_jobs']
                )

                # Registrar todas las métricas promedio de CV
                metrics_to_log = {}
                for metric_name in cv_results.keys():
                    if 'test_' in metric_name:
                        key = f"avg_cv_{metric_name.replace('test_', '')}"
                        metrics_to_log[key] = np.mean(cv_results[metric_name])
                        
                        key_std = f"std_cv_{metric_name.replace('test_', '')}"
                        metrics_to_log[key_std] = np.std(cv_results[metric_name])

                mlflow.log_metrics(metrics_to_log)
                logger.info(f"Métricas registradas. (avg_cv_gmean: {metrics_to_log.get('avg_cv_gmean', 0):.4f})")
                
                
                # --- 5. FIRMA Y REGISTRO DEL MODELO ---
                logger.info("Generando firma del modelo...")
                # Usamos el 'best_pipeline' para la firma
                y_pred_signature = best_pipeline.predict(X_train)
                signature = infer_signature(X_train, y_pred_signature)

                logger.info("Registrando el *mejor modelo* en MLflow...")
                mlflow.sklearn.log_model(
                    sk_model=best_pipeline, # ¡Registramos el mejor estimador!
                    artifact_path="model_pipeline", 
                    signature=signature,
                    input_example=X_train.iloc[:5],
                    registered_model_name=f"{model_name}_model"
                )
                
                logger.info(f"Experimento {model_name} (GridSearch) finalizado y registrado.")

        except Exception as e:
            logger.error(f"Error en el experimento {model_name}: {e}", exc_info=True)
            mlflow.end_run(status="FAILED")

    logger.info("--- Todos los experimentos han finalizado ---")
    logger.info(f"Ejecuta 'mlflow ui' en tu terminal para ver los resultados.")

# -----------------------------------------------------------------
# PUNTO DE ENTRADA DEL SCRIPT
# -----------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de entrenamiento y optimización de modelos (GridSearch) para el proyecto de riesgo crediticio.")
    parser.add_argument('--config', type=str, default='params.yaml', help='Ruta al archivo de configuración YAML.')
    
    args = parser.parse_args()
    main(config_path=args.config)