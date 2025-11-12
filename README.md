Proyecto MLOps – Clasificación de Riesgo Crediticio
Descripción general

Este proyecto implementa un flujo completo de Machine Learning Operations (MLOps) para la clasificación del riesgo crediticio a partir del South German Credit Dataset.
El sistema automatiza todas las etapas del ciclo de vida de un modelo de machine learning: desde la ingesta y preparación de datos, hasta el entrenamiento, optimización, registro y despliegue del modelo final.

El objetivo principal es desarrollar una arquitectura modular, reproducible y escalable, basada en buenas prácticas.
El flujo de trabajo está controlado mediante DVC (Data Version Control) y MLflow, integrando versionado de datos, trazabilidad de experimentos y gestión de modelos bajo un mismo entorno.

Base de datos utilizada

El proyecto utiliza el conjunto de datos South German Credit Dataset, un registro de solicitudes de crédito con variables demográficas, laborales y financieras de los solicitantes.
La base de datos original está en formato CSV y fue convertida a Parquet para optimizar la lectura y el almacenamiento.
Durante el proceso se aplicaron técnicas de limpieza, codificación de variables categóricas y balanceo de clases mediante SMOTE.

Arquitectura y estructura del proyecto

El repositorio sigue la plantilla Cookiecutter Data Science, lo que garantiza una organización clara y mantenible del código, los datos y los artefactos generados.

├── data/
│   ├── raw/                  # Datos originales (CSV)
│   ├── processed/            # Datos transformados (Parquet)
│
├── notebooks/                # Notebooks de exploración y análisis
│
├── src/                      # Código fuente modular
│   ├── data/                 # Módulos de ingesta y preparación de datos
│   ├── features/             # Generación y selección de características
│   ├── models/               # Entrenamiento, validación y evaluación
│   ├── utils/                # Funciones auxiliares (logging, métricas, etc.)
│   └── main.py               # Punto de entrada principal del pipeline
│
├── models/                   # Modelos entrenados y exportados (.joblib)
│
├── reports/                  # Visualizaciones, resultados y documentación
│
├── config.py                 # Configuración global del proyecto
├── params.yaml               # Parámetros de entrenamiento y rutas
├── requirements.txt          # Dependencias del entorno
├── dvc.yaml                  # Definición del pipeline DVC
└── README.md                 # Descripción general del proyecto

Flujo general del programa

El programa automatiza la ejecución completa del pipeline MLOps, integrando las siguientes etapas:

Ingesta y preparación de datos

Lectura del archivo CSV y conversión a Parquet.

Limpieza, codificación y tratamiento de valores faltantes.

Balanceo de clases con SMOTE.

Entrenamiento y optimización del modelo

Entrenamiento de múltiples algoritmos: Logistic Regression, Random Forest, KNN, XGBoost, SVC, MLP.

Búsqueda de hiperparámetros con GridSearchCV y registro automático en MLflow.

Detección de overfitting y underfitting mediante validación cruzada.

Gestión y registro de experimentos

Uso de MLflow Tracking Server (desplegado en Google Cloud) para registrar parámetros, métricas y artefactos.

Almacenamiento de versiones de modelos en el MLflow Model Registry.

Exportación del mejor modelo en formato .joblib.

Despliegue y monitoreo

Implementación de una API con FastAPI para realizar predicciones.

Empaquetado en contenedores con Docker.

Preparación para monitoreo de rendimiento y detección de data drift.

Requerimientos

El proyecto utiliza Python 3.12.12

Instalar todas las dependencias con:

pip install -r requirements.txt

Ejecución del proyecto
1. Clonar el repositorio
git clone https://github.com/<usuario>/<nombre-del-repositorio>.git
cd <nombre-del-repositorio>

2. Ejecutar el pipeline completo
python src/main.py

3. Reproducir las etapas con DVC
dvc repro

4. Consultar los experimentos registrados

Acceder al servidor remoto de MLflow configurado en Google Cloud:

[http://<tu-servidor-mlflow>](https://mlflow-super-g57-137680020436.us-central1.run.app/)

5. Desplegar la API de predicción


Resultados

El proyecto produce un modelo final optimizado para la clasificación del riesgo crediticio, validado bajo métricas como Accuracy, Precision, Recall, F1-score, AUC, G-Mean y Matriz de confusión.
El modelo es reproducible, versionado y desplegable, siguiendo las mejores prácticas del ciclo MLOps, con trazabilidad total de los datos, experimentos y resultados.
