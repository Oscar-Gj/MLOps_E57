
# ====================================================
# LIBRO DE LIBRERÍAS - PROYECTO CREDIT RISK ML
# Autor: Equipo 57 MLOps
# Descripción: Importaciones centralizadas para EDA,
# preprocesamiento, modelado, evaluación y despliegue.
# ====================================================

# =============================
# Núcleo y manejo de datos
# =============================
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import yaml
from dotenv import load_dotenv

# =============================
# Visualización y análisis exploratorio (EDA)
# =============================
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# =============================
# Preprocesamiento y utilidades
# =============================
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import FunctionTransformer
# =============================
# Modelado y algoritmos
# =============================
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# =============================
# Balanceo de clases
# =============================
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# =============================
# Versionado y trazabilidad (opcional para Fase 4+)
# =============================
import mlflow
from mlflow import MlflowClient
import dvc.api
# ⚠️ Estos se usan en Fase 5, se dejan comentados para evitar errores
# from evidently.report import Report
# from evidently.metric_preset import DataDriftPreset
# from evidently import ColumnMapping
# from deepchecks.tabular import Dataset, Suite
# from fairlearn.metrics import MetricFrame, selection_rate

# =============================
# Explicabilidad (opcional)
# =============================
import shap
import lime
import lime.lime_tabular

# =============================
# Utilidades adicionales
# =============================
from tqdm import tqdm
import joblib
import logging
import warnings
warnings.filterwarnings("ignore")

# =============================
# Configuración inicial
# =============================
load_dotenv()  # Carga variables de entorno (.env si existe)

# =============================
# Helper: Configurar logging global
# =============================
def get_logger(name: str):
    """Configura un logger con formato estandarizado."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
