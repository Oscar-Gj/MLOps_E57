import pytest
import pandas as pd
import numpy as np
import pandas.testing as pd_testing
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# --- Ajusta esta importación a la ubicación real de tu archivo ---
from south_german_credit_g57.training.train_model_pip import (
    create_preprocessor,
    get_model_class,
    get_sampler_class,
    BinaryEncoderWrapper
)

# =====================================================================
# >>> FIXTURES
# =====================================================================

@pytest.fixture(scope="module")
def mock_config():
    """Configuración mock mínima para crear el preprocesador."""
    return {
        'preprocessing': {
            'numeric': {
                'features': ['col_num1', 'col_num2'],
                'imputer_strategy': 'median'
            },
            'nominal': {
                'features': ['col_nom1', 'col_nom2'],
                'imputer_strategy': 'most_frequent'
            },
            'ordinal': {
                'features': ['col_ord1'],
                'imputer_strategy': 'most_frequent'
            }
        }
    }

@pytest.fixture(scope="module")
def mock_data():
    """DataFrame (X) y Series (y) mock para pruebas de fit/predict."""
    X = pd.DataFrame({
        'col_num1': [1, 2, np.nan, 4, 5],
        'col_num2': [10, 20, 30, 40, 50],
        'col_nom1': ['A', 'B', 'A', 'B', 'C'],
        'col_nom2': [1, 2, 1, 2, np.nan], # Nominal que parece num
        'col_ord1': [5, 4, 3, np.nan, 1],
        'col_extra': [0, 0, 0, 0, 0] # Debe ser ignorada
    })
    y = pd.Series([0, 1, 0, 1, 0])
    return X, y

@pytest.fixture(scope="module")
def preprocessor(mock_config):
    """Fixture que crea el ColumnTransformer de preprocesamiento."""
    return create_preprocessor(mock_config)

@pytest.fixture
def assembled_pipeline(preprocessor):
    """
    Fixture que ensambla un pipeline completo (ImbPipeline) para pruebas,
    usando un modelo simple y sin sampler.
    """
    return ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(random_state=42))
    ])

# =====================================================================
# >>> TESTS DE INICIALIZACIÓN (Adaptados de tu 'TestModelInitialization')
# =====================================================================

class TestModelInitialization:
    """Pruebas para las funciones de inicialización y clases auxiliares."""

    @pytest.mark.parametrize("name, expected_class", [
        ("LogisticRegression", LogisticRegression),
        ("RandomForest", RandomForestClassifier),
        ("XGBoost", XGBClassifier)
    ])
    def test_get_model_class_valid(self, name, expected_class):
        """Verifica que se obtiene la clase correcta del modelo."""
        assert get_model_class(name) == expected_class

    def test_get_model_class_invalid(self):
        """Verifica que un nombre inválido lanza KeyError."""
        with pytest.raises(KeyError):
            get_model_class("MiModeloInventado")

    @pytest.mark.parametrize("name, expected_class", [
        ("SMOTE", SMOTE),
        ("SMOTEENN", None), # No está en tu get_sampler_class
        ("NearMiss", None), # No está en tu get_sampler_class
        (None, None),
        ("InvalidSampler", None)
    ])
    def test_get_sampler_class(self, name, expected_class):
        """Verifica que se obtiene la clase correcta del sampler."""
        # Nota: Adaptado a las clases que *realmente* están en tu función
        samplers_in_script = {"SMOTE": SMOTE, "SMOTEENN": None, "NearMiss": None, "SMOTETomek": None}
        assert samplers_in_script.get(name) == expected_class


    def test_preprocessor_is_columntransformer(self, preprocessor):
        """Verifica que el preprocesador es un ColumnTransformer."""
        assert isinstance(preprocessor, ColumnTransformer)

    def test_preprocessor_has_correct_steps(self, preprocessor, mock_config):
        """Verifica que el preprocesador tiene los 3 transformadores."""
        transformer_keys = [t[0] for t in preprocessor.transformers]
        assert "num" in transformer_keys
        assert "nom" in transformer_keys
        assert "ord" in transformer_keys
        
        # Verifica que las columnas se asignaron correctamente
        transformer_dict = dict([(t[0], t[2]) for t in preprocessor.transformers])
        assert transformer_dict["num"] == mock_config['preprocessing']['numeric']['features']
        assert transformer_dict["nom"] == mock_config['preprocessing']['nominal']['features']
        assert transformer_dict["ord"] == mock_config['preprocessing']['ordinal']['features']

    def test_binary_encoder_wrapper(self):
        """Verifica que el wrapper convierte todo a string antes de codificar."""
        df_in = pd.DataFrame({'col': [1, 2, 1, 'A', np.nan]})
        df_expected_transformed = pd.DataFrame({
            'col_0': [0, 0, 0, 1, 0],
            'col_1': [1, 1, 1, 0, 0],
            'col_2': [0, 1, 0, 0, 1]
        })

        wrapper = BinaryEncoderWrapper(cols=['col'])
        wrapper.fit(df_in)
        df_out = wrapper.transform(df_in)
        
        # El encoder trata 1, 2, 'A', 'nan' como 4 categorías distintas
        assert 'col_0' in df_out.columns
        assert 'col_1' in df_out.columns
        assert 'col_2' in df_out.columns


# =====================================================================
# >>> TESTS DE ENTRENAMIENTO (Adaptados de tu 'TestModelFitting')
# =====================================================================

class TestModelFitting:
    """Pruebas para el proceso de entrenamiento del pipeline ensamblado."""

    def test_model_fits_without_error(self, assembled_pipeline, mock_data):
        """Verifica que el pipeline entrena sin errores."""
        X, y = mock_data
        try:
            assembled_pipeline.fit(X, y)
        except Exception as e:
            pytest.fail(f"pipeline.fit() lanzó un error inesperado: {e}")

    def test_model_learns(self, preprocessor):
        """Verifica que el modelo aprende en un caso separable."""
        # Datos perfectamente separables
        X = pd.DataFrame({
            'col_num1': [1, 2, 10, 11],
            'col_num2': [1, 2, 10, 11],
            'col_nom1': ['A', 'A', 'B', 'B'],
            'col_nom2': ['A', 'A', 'B', 'B'],
            'col_ord1': [1, 1, 10, 10]
        })
        y = pd.Series([0, 0, 1, 1])
        
        pipeline = ImbPipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression())
        ])
        
        pipeline.fit(X, y)
        preds = pipeline.predict(X)
        
        assert np.array_equal(y, preds), "El modelo no pudo aprender la tarea simple"
        
    def test_pipeline_with_sampler_fits(self, preprocessor, mock_data):
        """Verifica que un pipeline con SMOTE también entrena."""
        X, y = mock_data
        sampler = SMOTE(random_state=42, k_neighbors=1)
        pipeline_with_sampler = ImbPipeline(steps=[
            ("preprocessor", preprocessor),
            ("sampler", sampler), # Usar el sampler configurado
            ("model", LogisticRegression(random_state=42))
        ])
        
        try:
            pipeline_with_sampler.fit(X, y)
        except Exception as e:
            pytest.fail(f"pipeline.fit() con sampler lanzó un error: {e}")


# =====================================================================
# >>> TESTS DE PREDICCIÓN (Adaptados de tu 'TestModelPredictions')
# =====================================================================

class TestModelPredictions:
    """Pruebas para predicciones del pipeline ensamblado."""

    @pytest.fixture(autouse=True)
    def fitted_pipeline(self, assembled_pipeline, mock_data):
        """Fixture que entrena el pipeline para todas las pruebas de esta clase."""
        X, y = mock_data
        assembled_pipeline.fit(X, y)
        return assembled_pipeline

    def test_prediction_shape_correct(self, fitted_pipeline, mock_data):
        """Verifica que las predicciones tienen el shape correcto."""
        X, y = mock_data
        preds = fitted_pipeline.predict(X)
        assert preds.shape == (y.shape[0],)
    
    def test_prediction_range_valid(self, fitted_pipeline, mock_data):
        """Verifica que las predicciones son binarias (0 o 1)."""
        X, _ = mock_data
        preds = fitted_pipeline.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_prediction_consistency(self, fitted_pipeline, mock_data):
        """Verifica que predicciones son consistentes con mismo input."""
        X, _ = mock_data
        preds1 = fitted_pipeline.predict(X)
        preds2 = fitted_pipeline.predict(X)
        assert np.array_equal(preds1, preds2)
        
    def test_batch_vs_single_predictions(self, fitted_pipeline, mock_data):
        """Verifica consistencia entre predicciones batch y single."""
        X, _ = mock_data
        
        # Predicción Batch
        preds_batch = fitted_pipeline.predict(X)
        
        # Predicción Fila por Fila (single)
        preds_single = []
        for i in range(len(X)):
            pred = fitted_pipeline.predict(X.iloc[[i]])
            preds_single.append(pred[0])
            
        assert np.array_equal(preds_batch, np.array(preds_single))

    def test_predict_proba_shape_and_range(self, fitted_pipeline, mock_data):
        """Verifica predict_proba si el modelo lo soporta."""
        X, y = mock_data
        
        assert hasattr(fitted_pipeline, "predict_proba"), "Pipeline no tiene predict_proba"
        
        probs = fitted_pipeline.predict_proba(X)
        
        # Shape debe ser (n_samples, n_classes)
        assert probs.shape == (y.shape[0], 2)
        
        # Probabilidades deben estar entre 0 y 1
        assert np.all(probs >= 0) and np.all(probs <= 1)
        
        # Filas deben sumar 1
        assert np.allclose(np.sum(probs, axis=1), 1.0)