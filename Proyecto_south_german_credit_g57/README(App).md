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
