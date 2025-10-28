# =========================================================
# LIBRERÍAS BASE — Módulo compartido
# =========================================================

# --- Estándar ---
import os
import sys
import warnings
from pathlib import Path

# --- Cálculo y datos ---
import numpy as np
import pandas as pd

# --- Visualización ---
import matplotlib.pyplot as plt
import seaborn as sns

# --- Scikit-learn (preprocesamiento, evaluación, modelos base) ---
from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
    cross_val_predict
)
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    OrdinalEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    make_scorer
)

# --- Modelos frecuentes ---
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# --- Imbalanced-learn ---
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.metrics import geometric_mean_score

# --- XGBoost (si está instalado) ---
try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

# --- Configuración global ---
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# --- Warnings y estilo ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="This Pipeline instance is not fitted yet")
sns.set_style("whitegrid")

print("✅ Librerías cargadas y entorno base inicializado correctamente.")
