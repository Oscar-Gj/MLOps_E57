# ==========================================================
# CONFIGURACIÓN GLOBAL DEL PROYECTO - SOUTH GERMAN CREDIT G57
# ==========================================================
"""
Archivo: params.yaml
Propósito:
Define todos los parámetros del pipeline MLOps del proyecto *South German Credit G57*.
Centraliza la configuración de datos, limpieza, preprocesamiento, modelado y evaluación.
Este archivo permite mantener la reproducibilidad y flexibilidad en cada fase del flujo.

-------------------------------------------------------------
OBJETIVO GENERAL
-------------------------------------------------------------
Servir como único punto de configuración del pipeline completo:
    - Fase 1: Limpieza y validación de datos
    - Fase 2: Construcción del pipeline (preprocesamiento)
    - Fase 3: Entrenamiento de modelos y registro en MLflow
    - Fase 4: Evaluación y generación de reportes

Cualquier cambio en los parámetros de entrenamiento o datos puede realizarse
desde este archivo sin modificar el código Python, promoviendo buenas prácticas
de ingeniería MLOps.

-------------------------------------------------------------
ESTRUCTURA GENERAL
-------------------------------------------------------------
1. base              → Parámetros generales y columna objetivo.
2. data              → Rutas de acceso a los datasets (raw, processed, train, test).
3. data_cleaning     → Reglas de limpieza, tratamiento de outliers y rare values.
4. preprocessing     → Definición de features numéricos, categóricos y ordinales.
5. grid_search       → Configuración de la búsqueda de hiperparámetros (CV).
6. training          → Modelos, técnicas de muestreo y grids de parámetros.
7. mlflow            → Conexión local o cloud (seguimiento de experimentos).
8. reports           → Rutas y configuración de salida de métricas y visualizaciones.

-------------------------------------------------------------
MODO DE EJECUCIÓN
-------------------------------------------------------------
Este archivo es leído directamente por `main.py` a través de:
    config = yaml.safe_load(open("params.yaml", "r"))

Ejemplo de ejecución local:
    python -m south_german_credit_g57.main --config ../params.yaml --full-eval

Ejemplo de ejecución en Cloud (MLflow remoto):
    python -m south_german_credit_g57.main --config ../params.yaml --full-eval

-------------------------------------------------------------
CONFIGURACIÓN CLOUD
-------------------------------------------------------------
Si el modo seleccionado es "cloud", el experimento será registrado en un
servidor remoto de MLflow desplegado en Google Cloud Run o Vertex AI.

Antes de ejecutar en este modo:
    1️⃣ Solicitar acceso al **administrador del entorno Cloud**.
    2️⃣ Tener credenciales activas de Google Cloud.
    3️⃣ Autenticarse en el proyecto correspondiente:
         gcloud auth login
         gcloud config set project laboratorio1-447417
    4️⃣ Verificar la URI en esta sección:

        mlflow:
          mode: "cloud"
          tracking_uri: "https://mlflow-super-g57-137680020436.us-central1.run.app"
          experiment_name: "Experimento-Conexión-MLFlow-Grupo57"

⚠️ Si el usuario no tiene permisos, el sistema devolverá errores como:
   "The caller does not have permission" o "Access denied".

-------------------------------------------------------------
DETALLES DE CADA SECCIÓN
-------------------------------------------------------------
🔹 base:
    - Define el seed (`random_state`) y la variable objetivo (`target_col`).
    - Garantiza consistencia entre fases (train/test split y modelado).

🔹 data:
    - Rutas absolutas o relativas de los datasets.
    - Las rutas pueden adaptarse al entorno local o al entorno Cloud Storage.

🔹 data_cleaning:
    - Define columnas a renombrar, eliminar o imputar.
    - Controla valores atípicos (`outlier_cols`) y categorías raras (`rare_cols`).
    - Permite ajustar la proporción de test (`test_size`).

🔹 preprocessing:
    - Separa las features por tipo: numéricas, nominales y ordinales.
    - Define estrategias de imputación por tipo de variable.
    - Compatible con Scikit-Learn `ColumnTransformer`.

🔹 grid_search:
    - Configura la validación cruzada (CV y repeticiones).
    - Define la métrica principal de optimización (ej. ROC AUC).
    - Permite paralelizar en todos los cores disponibles (`n_jobs: -1`).

🔹 training:
    - Lista los modelos a entrenar (LogisticRegression, RandomForest, XGBoost, etc.).
    - Cada modelo define su `param_grid` y técnica de balanceo (SMOTE, NearMiss).
    - Los parámetros se aplican automáticamente desde `train_model_pip.py`.

🔹 mlflow:
    - Determina si se ejecuta en modo local o cloud.
    - `experiment_name` y `tracking_uri` son usados para el registro de runs.
    - `evaluation_experiment_name` controla el registro de métricas finales.

🔹 reports:
    - Carpeta de salida donde se guardan las gráficas y reportes generados.
    - Compatible con los artefactos de MLflow (plots, .txt, .png, .html).

-------------------------------------------------------------
AUTORÍA Y CONTROL DE VERSIONES
-------------------------------------------------------------
Autor: Equipo 57 MLOps
Fecha: Noviembre 2025
Versión: 2.0
Compatibilidad: Python 3.12 / 3.13 (Windows, macOS ARM, Linux)
-------------------------------------------------------------
"""
