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