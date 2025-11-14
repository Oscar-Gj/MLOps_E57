import pytest
import os
import shutil
import pandas as pd
import yaml
from pathlib import Path
from unittest.mock import MagicMock

from south_german_credit_g57.main import run_pipeline

# --- Definición de Rutas para el Test ---
PROJECT_ROOT = Path(__file__).resolve().parents[2] 
TEST_CONFIG_PATH = str(PROJECT_ROOT / "tests/integration/test_params.yaml")
OUTPUT_DIR = PROJECT_ROOT / "tests/outputs"
FALLBACK_MODEL_DIR = PROJECT_ROOT / "models_fallback"
REQUIREMENTS_MARKER = PROJECT_ROOT / ".requirements_verified"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

# =====================================================================
# >>> FIXTURES DE CONFIGURACIÓN
# =====================================================================

@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown_files():
    """
    Fixture (módulo) que se ejecuta ANTES de todas las pruebas en este archivo
    y DESPUÉS de todas ellas. Se encarga de limpiar los artefactos generados.
    """
    # --- SETUP ---
    # Limpiar directorios de salidas de pruebas anteriores
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    if FALLBACK_MODEL_DIR.exists():
        shutil.rmtree(FALLBACK_MODEL_DIR)
    if REQUIREMENTS_MARKER.exists():
        os.remove(REQUIREMENTS_MARKER)
    if MLRUNS_DIR.exists():
        shutil.rmtree(MLRUNS_DIR)
    # Crear el directorio de salida vacío
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- YIELD ---
    yield

    # --- TEARDOWN ---
    # Limpiar los directorios de salida después de las pruebas
    print("\n--- Limpiando directorios de prueba ---")
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    if FALLBACK_MODEL_DIR.exists():
        shutil.rmtree(FALLBACK_MODEL_DIR)
    if REQUIREMENTS_MARKER.exists():
        os.remove(REQUIREMENTS_MARKER)
    if MLRUNS_DIR.exists():
        shutil.rmtree(MLRUNS_DIR)

@pytest.fixture(autouse=True)
def mock_external_services(mocker):
    """
    Fixture (función) que se ejecuta para CADA prueba.
    Mocks (reemplaza) todas las llamadas a servicios externos (pip, MLflow).
    """
    
    # 1. Mock de la instalación de dependencias (no queremos que instale nada)
    mocker.patch("south_german_credit_g57.main.verify_and_install_requirements",
                return_value=None)
    
    # # 2. Mock de todas las llamadas a MLflow
    # mocker.patch("mlflow.set_tracking_uri", return_value=None)
    # mocker.patch("mlflow.set_experiment", return_value=None)
    # mocker.patch("mlflow.end_run", return_value=None)
    
    # # Mockear el context manager 'mlflow.start_run'
    # mock_run = MagicMock()
    # mock_context_manager = MagicMock(__enter__=MagicMock(return_value=mock_run),
    #                                 __exit__=MagicMock(return_value=None))
    # mocker.patch("mlflow.start_run", return_value=mock_context_manager)
    
    # # Mockear el logging de métricas y parámetros
    # mocker.patch("mlflow.log_param", return_value=None)
    # mocker.patch("mlflow.log_metrics", return_value=None)
    # mocker.patch("mlflow.log_artifact", return_value=None)
    
    # # Mockear 'search_runs' (para eval_main) para que devuelva un modelo falso
    # fake_run_df = pd.DataFrame({
    #     "tags.mlflow.runName": ["LogisticRegression_GridSearch"],
    # })
    # mocker.patch("mlflow.search_runs", return_value=fake_run_df)
    
    # 3. Mockear el guardado y carga de modelos
    
    # Forzamos a 'log_model' a fallar para probar el *fallback*
    # Esto simula un servidor MLflow caído
    mocker.patch("mlflow.sklearn.log_model", 
                side_effect=Exception("Mock: Falla de conexión con el Registry de MLflow"))
    fake_run_df = pd.DataFrame({
        "tags.mlflow.runName": ["LogisticRegression_GridSearch"],
    })
    mocker.patch("south_german_credit_g57.evaluation.metrics.mlflow.search_runs", fake_run_df)
    mocker.patch("south_german_credit_g57.evaluation.metrics.mlflow.sklearn.load_model", 
                side_effect=Exception("Mock: No se puede cargar desde 'models:/' en un test"))
    # Mockeamos 'load_model' (para eval_main)
    # Aquí tenemos un problema: eval_main espera cargar un modelo
    # que train_main guardó.
    # Vamos a dejar que 'train_main' falle al loggear y guarde localmente.
    # Y 'eval_main' fallará porque no puede cargar desde 'models:/...'.
    # Por ahora, probaremos el pipeline *sin* 'eval_main' (full_eval=False)
    # que es el flujo Carga -> Clean -> Train -> Save Model.
    # mocker.patch("mlflow.sklearn.load_model", 
    #             side_effect=Exception("Mock: No se puede cargar desde 'models:/' en un test"))


# =====================================================================
# >>> TESTS DE INTEGRACIÓN
# =====================================================================

class TestIntegrationPipeline:
    
    def test_clean_and_train_pipeline_flow(self):
        """
        Prueba el flujo: Carga -> Clean -> Train -> Save Model (Fallback).
        No ejecuta la evaluación final (ya que depende de MLflow Registry).
        """
        
        # --- Arrange ---
        # Las fixtures 'setup_and_teardown_files' y 'mock_external_services'
        # ya se ejecutaron automáticamente.
        
        # --- Act ---
        # Ejecutamos el pipeline con el config de prueba
        # 'full_eval=False' es clave aquí.
        try:
            run_pipeline(
                config_path=TEST_CONFIG_PATH,
                skip_clean=False,
                skip_train=False,
                full_eval=False 
            )
        except Exception as e:
            pytest.fail(f"La ejecución del pipeline falló inesperadamente: {e}")
            
        # --- Assert ---
        # Verificamos que los artefactos esperados fueron creados en 'tests/outputs'
        
        # 1. ¿Se crearon los datos limpios?
        assert (OUTPUT_DIR / "processed.csv").exists(), "El archivo 'processed.csv' no fue creado."
        assert (OUTPUT_DIR / "train.csv").exists(), "El archivo 'train.csv' no fue creado."
        assert (OUTPUT_DIR / "test.csv").exists(), "El archivo 'test.csv' no fue creado."
        
        # 2. ¿Los datos de entrenamiento tienen contenido?
        df_train = pd.read_csv(OUTPUT_DIR / "train.csv")
        assert df_train.shape[0] > 0, "El 'train.csv' está vacío."
        
        # 3. ¿Se guardó el modelo en el 'fallback' (ya que MLflow fue mockeado)?
        assert FALLBACK_MODEL_DIR.exists(), "El directorio 'models_fallback' no fue creado."
        
        # Buscamos el modelo guardado (LogisticRegression)
        model_dirs = list(FALLBACK_MODEL_DIR.glob("*_fallback"))
        assert len(model_dirs) > 0, "No se encontró ningún modelo en 'models_fallback'."
        
        # Verificamos que el archivo .pkl existe
        model_path = model_dirs[0] / "model.pkl"
        assert model_path.exists(), "El archivo 'model.pkl' no fue guardado en el fallback."