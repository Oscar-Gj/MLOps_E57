# CONFIGURACIÓN GLOBAL DEL PROYECTO - SOUTH GERMAN CREDIT G57

Archivo: **params.yaml**  
Propósito:  
Define todos los parámetros del pipeline MLOps del proyecto *South German Credit G57*.  
Centraliza la configuración de datos, limpieza, preprocesamiento, modelado y evaluación. Este archivo permite mantener la reproducibilidad y flexibilidad en cada fase del flujo.

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

"""



# MAIN PIPELINE ORCHESTRATOR - SOUTH GERMAN CREDIT G57

"""
Módulo principal encargado de orquestar todas las fases del pipeline MLOps
del proyecto *South German Credit G57*. Coordina la ejecución completa del flujo
de Machine Learning bajo un enfoque reproducible, modular y automatizado, tanto
en entornos locales como en la nube (Cloud).

-------------------------------------------------------------
OBJETIVO GENERAL
-------------------------------------------------------------
Automatizar el ciclo de vida del modelo de crédito:
    1. Limpieza y validación de datos
    2. Construcción y entrenamiento del pipeline
    3. Evaluación y registro de métricas (MLflow)
    4. Evaluación extendida (opcional, controlada por flags)

Incluye además la verificación de dependencias (requirements.txt)
y su instalación automática solo la primera vez.

-------------------------------------------------------------
ARQUITECTURA DEL PIPELINE
-------------------------------------------------------------
            ┌──────────────────┐
            │  clean_data.py   │  → Limpieza, imputación y validación de datos
            └────────▲─────────┘
                     │
 metrics.py  ◄── main.py ───► pipeline.py
                     │
            ┌────────▼────────┐
            │ train_model_pip │  → Entrenamiento, validación y registro en MLflow
            └────────▲────────┘
                     │
                 logger.py

-------------------------------------------------------------
MODOS DE EJECUCIÓN
-------------------------------------------------------------
**1️⃣ Ejecución local (por defecto):**
Guarda experimentos y modelos dentro del proyecto, bajo la carpeta:
    ./mlruns

Ejemplo:
    python -m south_german_credit_g57.main --config ../params.yaml --full-eval

**2️⃣ Ejecución en entorno Cloud (GCP / MLflow remoto):**
Permite registrar los experimentos, métricas y modelos directamente
en un servidor remoto (por ejemplo, MLflow desplegado en Google Cloud Run).

Antes de usar este modo, debes:
    - Solicitar acceso al administrador del entorno Cloud.
    - Tener credenciales activas de Google Cloud (cuenta @tec.mx o institucional).
    - Estar autenticado en el proyecto autorizado de GCP con el comando:

        gcloud auth login
        gcloud config set project <ID_PROYECTO_AUTORIZADO>

    - Verificar que el `tracking_uri` en `params.yaml` apunte a la URL del servidor MLflow remoto:

        mlflow:
          tracking_uri: "https://mlflow-super-g57-137680020436.us-central1.run.app"
          experiment_name: "Experimento-Conexión-MLFlow-Grupo57"

Ejemplo de ejecución en modo Cloud:
    python -m south_german_credit_g57.main --config ../params.yaml --full-eval

⚠️ Importante:
El acceso remoto está restringido. Si el usuario no tiene permisos,
recibirá un error de tipo "Permission denied" o "Caller does not have permission".
Debe solicitar autorización al **Administrador del Cloud del proyecto** antes de reintentar.

-------------------------------------------------------------
REQUERIMIENTOS
-------------------------------------------------------------
- Python 3.12 o 3.13
- Entorno virtual activado (p. ej. `jarvis`)
- Archivo de configuración `params.yaml`
- Archivo de dependencias `requirements.txt` actualizado
- Acceso autorizado al servidor MLflow (para modo Cloud)
- Conectividad estable (si se ejecuta remotamente)

-------------------------------------------------------------
FLAGS DISPONIBLES
-------------------------------------------------------------
--config       → Ruta al archivo YAML de configuración.
--skip-clean   → Omitir la fase de limpieza de datos.
--skip-train   → Omitir el entrenamiento del modelo.
--skip-eval    → Omitir la evaluación final.
--full-eval    → Ejecuta la evaluación extendida al final del pipeline.

Ejemplo completo:
    python -m south_german_credit_g57.main --config ../params.yaml --full-eval

-------------------------------------------------------------
RESULTADOS Y SALIDAS
-------------------------------------------------------------
✔ Datos procesados → data/processed/
✔ Modelos entrenados → models/
✔ Métricas → reports/metrics/
✔ Experimentos MLflow → mlruns/ o servidor remoto (Cloud)
✔ Logs → logs/YYYY-MM-DD.log

"""



# Pruebas (Testing)

Este proyecto utiliza `pytest` para asegurar la calidad, correctitud y robustez del código en todas las fases del pipeline de MLOps.

## 1. Configuración del Entorno de Pruebas

Antes de ejecutar las pruebas, asegúrate de tener el entorno virtual activado y las dependencias principales instaladas:
```bash
pip install -r requirements.txt
```

Unicamente si deseas generar un reporte con formato HTML, instala la dependencia `pytest-cov`:
```bash
pip install pytest-cov
```

## 2. Ejecución de Pruebas

Todas las pruebas se encuentran en el directorio `Proyecto_south_german_credit_g57/tests/`.

### Ejecución Completa (Modo Detallado)

Para ejecutar el conjunto completo de pruebas (unitarias y de integración) y ver un desglose detallado de cada test, usa el comando `pytest` con el flag `-v` (verbose):
```bash
pytest -v
```

### Ejecución Rápida (Modo Silencioso)

Para cumplir con los requisitos del proyecto (T3), puedes ejecutar las pruebas en "modo silencioso" (`-q` / `quiet`). Este comando solo mostrará el resultado final (ej. `25 passed, 3 skipped in 140s`), lo cual es ideal para logs limpios o flujos de CI/CD.

Ejecuta este comando desde la raíz del proyecto:
```bash
pytest -q
```

## 3. Cobertura de Pruebas (Opcional pero recomendado)

Para generar un reporte de "cobertura" (qué porcentaje de tu código fuente está cubierto por las pruebas), puedes usar `pytest-cov`.
```bash
# Ejecuta las pruebas y calcula la cobertura para la carpeta 'src'
pytest --cov=src
```

Para un reporte visual detallado en HTML:
```bash
# Genera un reporte HTML
pytest --cov=src --cov-report=html
```

Esto creará una carpeta `htmlcov/`. Abre el archivo `htmlcov/index.html` en tu navegador para ver línea por línea qué código fue probado y cuál no.

## 4. Estrategia de Pruebas Implementada

Se han implementado dos niveles de pruebas:

### Pruebas Unitarias (`tests/unit/`)

* **`test_preprocessing.py`**: Valida la lógica de `clean_data.py`.
* **`test_training.py`**: Valida la lógica de construcción de `train_model_pip.py` (creación de pipelines, manejo de samplers, etc.).
* **`test_metrics.py`**: Valida que el `metrics_module.py` calcule correctamente métricas clave (Accuracy, Precision, F1, ROC-AUC) usando casos de prueba definidos.

### Pruebas de Integración (`tests/integration/`)

* **`test_main_pipeline.py`**: Valida el flujo end-to-end del orquestador `main.py`. Esta prueba ejecuta el pipeline completo (Clean → Train) usando datos de prueba (`tests/fixtures/`) y una configuración temporal (`tests/integration/test_params.yaml`).
* Utiliza `pytest-mock` para simular fallos de servicios externos (como `mlflow`), asegurando que la lógica de fallback (guardado local del modelo) funcione correctamente.

### Configuración (`pytest.ini`)

* Se ha configurado `pytest.ini` para añadir `src` al `PYTHONPATH` (evitando errores de importación) y para suprimir warnings informativos conocidos de `sklearn` y `mlflow`, resultando en una salida de pruebas limpia.

# 🤖 Simulador de Riesgo Crediticio (MLOps G57)

Este proyecto implementa una aplicación web completa para la predicción de riesgo crediticio, siguiendo un pipeline de MLOps desde el entrenamiento del modelo hasta su despliegue en un contenedor unificado.

La aplicación consta de dos componentes principales que se ejecutan en un solo contenedor Docker:

* **Backend (API de Inferencia):** Una API de FastAPI que sirve un modelo de Regresión Logística cargado directamente desde un Model Registry de MLflow.
* **Frontend (Interfaz de Usuario):** Una aplicación web interactiva de Streamlit que consume la API de FastAPI, permitiendo a los usuarios ingresar datos en un formulario amigable y recibir una predicción de riesgo.

---

## 🚀 Arquitectura de la Aplicación

Esta aplicación utiliza una arquitectura unificada dentro de un contenedor Docker, diseñada para ser portátil y fácil de desplegar.

* **Contenedor Docker:** Actúa como el servidor principal.
* **start.sh:** Un script de inicio que lanza ambos servicios.
* **API (FastAPI):** Se ejecuta en el puerto 8000. Al iniciar, se conecta a la URI de MLflow (`https://mlflow-super-g57...`) y descarga el modelo registrado (`LogisticRegression_model@best`).
* **UI (Streamlit):** Se ejecuta en el puerto 8001. Cuando un usuario envía el formulario, esta aplicación realiza una petición POST al backend de FastAPI en `http://127.0.0.1:8000/predict`.

---

## 📋 Características Principales

### API de Inferencia (Backend - FastAPI)

* **Endpoint /predict:** Recibe los 20 campos del formulario como un JSON, los convierte a un DataFrame de pandas y los pasa al modelo de MLflow.
* **Auto-documentación:** La API está completamente documentada con Swagger.
* **Validación de Datos:** Utiliza Pydantic para asegurar que los tipos de datos enviados a la API sean correctos (`float`).
* **Modelo desde MLflow:** Carga el modelo directamente desde el Model Registry de MLflow, asegurando que siempre se utilice la versión designada (`best`).

### Interfaz de Usuario (Frontend - Streamlit)

* **Formulario Amigable:** Traduce los 20 campos técnicos del modelo (ej. `credit_history`) a preguntas en español (ej. "Historial Crediticio") usando menús desplegables y sliders.
* **Visualización de Resultados:** Muestra la predicción final ("Riesgo Alto" / "Riesgo Bajo") con un indicador de confianza y una barra de progreso.
* **Interactivo:** Permite a los usuarios ajustar los valores y ver el impacto en la predicción.

---

## 🛠️ Prerrequisitos

Para ejecutar este proyecto, solo necesitas tener instalado y en ejecución:

* Docker
* Git (para clonar el repositorio)

---

## ⚡ Guía de Despliegue Rápido (Local)

Sigue estos pasos para construir y ejecutar la aplicación en tu máquina local.

### 1. Clonar el Repositorio

```
git clone https://github.com/Oscar-Gj/MLOps_E57.git
cd MLOps_E57
```

*(Nota: Reemplaza la URL si es diferente)*

### 2. Dar Permisos de Ejecución (Solo Linux/Mac)

Este paso es crucial para permitir que Docker ejecute el script de inicio.

```
chmod +x start.sh
```

*(Si estás en Windows, ejecuta este comando usando Git Bash)*

### 3. Construir la Imagen de Docker

Este comando leerá el Dockerfile, instalará las dependencias de `requirements.txt` (FastAPI, Streamlit, MLflow, etc.) y empaquetará tu aplicación.

```
docker build -t app-credito-g57:latest .
```

*(No olvides el "." al final)*

### 4. Ejecutar el Contenedor

Este comando inicia el contenedor y expone los puertos de la API y de la interfaz de usuario a tu máquina local.

```
docker run -p 8000:8000 -p 8001:8001 app-credito-g57:latest
```

---

## 🖥️ Cómo Usar la Aplicación

Una vez que el contenedor esté corriendo, tendrás acceso a los dos servicios:

### 1. Interfaz de Usuario (Streamlit)

Esta es la aplicación principal para usuarios finales.
**Acceso:** [http://127.0.0.1:8001](http://127.0.0.1:8001)

**Uso:**

* Verás un formulario con **20 campos**.
* Completa los campos usando los menús desplegables y sliders.
* Haz clic en el botón **"Predecir Riesgo"**.
* El resultado aparecerá en la parte inferior, mostrando la **predicción y la probabilidad**.

### 2. Documentación de la API (Swagger)

Si eres un desarrollador y quieres consumir la API directamente (por ejemplo, desde Postman o un script de Python), puedes usar la documentación de Swagger.
**Acceso:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

**Uso:**

* Verás el endpoint `POST /predict`.
* Ábrelo y haz clic en **"Try it out"**.
* Puedes usar el JSON de ejemplo (`schema_extra`) para enviar una petición de prueba.
* Haz clic en **"Execute"** para ver la respuesta JSON del modelo.

---

## 📁 Estructura de Archivos (Servidor)

```
├── app/
│   ├── main.py           # Lógica de la API (FastAPI)
│   ├── streamlit_app.py  # Lógica de la Interfaz (Streamlit)
│   └── a57.png           # Logo para la interfaz
│
├── Dockerfile            # Receta para construir el contenedor
├── requirements.txt      # Dependencias de Python (FastAPI, Streamlit, MLflow)
└── start.sh              # Script para iniciar ambos servicios
```

-------------------------------------------------------------
AUTORÍA Y CONTROL DE VERSIONES
-------------------------------------------------------------
Autor: Equipo 57 MLOps
Fecha: Noviembre 2025
Versión: 2.1
Compatibilidad: Python 3.12 / 3.13 | Windows, macOS ARM, Linux
-------------------------------------------------------------