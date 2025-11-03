# ==========================================================
# MAIN PIPELINE ORCHESTRATOR - SOUTH GERMAN CREDIT G57
# ==========================================================
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

-------------------------------------------------------------
AUTORÍA Y CONTROL DE VERSIONES
-------------------------------------------------------------
Autor: Equipo 57 MLOps
Fecha: Noviembre 2025
Versión: 2.1
Compatibilidad: Windows, macOS ARM, Linux
-------------------------------------------------------------
"""
