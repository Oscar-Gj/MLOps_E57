
# main.py
# -------------------------------------
# Orquestador principal del proyecto de Machine Learning.

# Este script ejecuta de manera automatizada el pipeline completo de modelado
# para el dataset South German Credit. Integra las etapas de preparación de datos,
# preprocesamiento, muestreo, modelado, evaluación y registro en MLflow.

# Autor: Equipo G57
# Versión: 1.0
###


# ==========================================================
#  Importaciones de librerías y módulos del proyecto
# ==========================================================
import warnings
warnings.filterwarnings("ignore")

from seed import set_seed, get_random_state
from prepare_data import load_dataset, prepare_dataset
from preprocessors import build_preprocessor
from pipeline_utils import train_and_log_model

from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

# ==========================================================
# Configuración general
# ==========================================================
# Establecer semilla global para reproducibilidad
set_seed()
RANDOM_STATE = get_random_state()

# Definir ruta del dataset Parquet
DATA_PATH = "data/03_df_eda_01.parquet"

# Nombre del experimento en MLflow
EXPERIMENT_NAME = "CreditRisk_MLOps_G57"

# ==========================================================
#  Carga y preparación del dataset
# ==========================================================
print("\n Cargando y preparando los datos...")
df = load_dataset(DATA_PATH)

# Definir columnas (las mismas que usaste en tus módulos)
num_cols = ["duration", "amount", "installment_rate", "present_residence", "age"]
cat_cols = ["credit_history", "purpose", "savings", "employment_duration", "other_debtors"]
ord_cols = ["personal_status_sex", "property", "housing", "job", "foreign_worker"]

X_train, X_test, y_train, y_test, X_all, y_all = prepare_dataset(
    df=df,
    target_col="credit_risk",
    num_cols=num_cols,
    cat_cols=cat_cols,
    ord_cols=ord_cols,
    test_size=0.3,
    random_state=RANDOM_STATE
)

print(" Datos cargados y divididos correctamente.")

# ==========================================================
#  Construcción del preprocesador
# ==========================================================
print("\n Construyendo preprocesador de variables...")
preprocessor = build_preprocessor(
    num_cols=num_cols,
    cat_cols=cat_cols,
    ord_cols=ord_cols
)
print(" Preprocesador listo.")

# ==========================================================
#  Definición del modelo y método de muestreo
# ==========================================================
print("\n Definiendo modelo y técnica de balanceo...")
model = LogisticRegression(
    solver="saga",
    max_iter=3000,
    penalty="l2",
    C=0.1,
    class_weight="balanced",
    random_state=RANDOM_STATE
)
sampler = SMOTE(random_state=RANDOM_STATE)
print(" Configuración de modelo y muestreo completada.")

# ==========================================================
#  Entrenamiento, validación y registro automático en MLflow
# ==========================================================
print("\n Iniciando entrenamiento y registro en MLflow...")

train_and_log_model(
    model=model,
    model_name="LogisticRegression_SMOTE",
    X=X_all,
    y=y_all,
    preprocessor=preprocessor,
    sampler=sampler,
    experiment_name=EXPERIMENT_NAME,
    random_state=RANDOM_STATE
)

print("\nEjecución completada con éxito.")
print(" Los resultados se registraron en MLflow correctamente.")
print(" Pipeline reproducible ejecutado sin errores.")
