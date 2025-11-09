import pytest
import pandas as pd
import numpy as np
import pandas.testing as pd_testing
from pathlib import Path

from south_german_credit_g57.preprocessing.clean_data import clean_data


# =====================================================================
# >>> PROPUESTA 1: FIXTURES PARAMETRIZADAS
# =====================================================================

@pytest.fixture(scope="module")
def mock_config():
    """Configuración mock para todas las pruebas."""
    return {
        'data_cleaning': {
            'rename_cols': ['col_num', 'col_str', 'col_outlier', 'col_rare', 'target', 'col_drop'],
            'drop_cols': ['col_drop'],
            'outlier_cols': {'col_outlier': 100}
        },
        'base': {'target_col': 'target'},
        'preprocessing': {
            'numeric': {'features': ['col_num', 'col_outlier']},
            'nominal': {'features': ['col_str']},
            'ordinal': {'features': ['col_rare']}
        }
    }


@pytest.fixture(scope="module")
def raw_dataframe():
    """DataFrame raw con todos los casos edge a probar."""
    return pd.DataFrame({
        'A_num': ['1', '2', '?', '4', '5', '6', '7', '8', '9'],
        'B_str': [' A ', 'B', 'NaN', 'B', 'B', 'B', 'B', 'B', 'B'],
        'C_out': [50, 150, 75, 'invalid', 90, 95, 110, 80, 85],
        'D_rare': ['rare1', 'cat1', 'cat1', 'cat1', 'cat1', 'cat1', 'cat1', 'cat1', 'cat1'],
        'E_target': [1.0, 0.0, 1.0, 'error', 0.0, 1.0, 1.0, 0.0, 1.0],
        'F_drop': ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x']
    })


@pytest.fixture(scope="module")
def clean_dataframe(raw_dataframe, mock_config):
    """Ejecuta clean_data una sola vez para todos los tests."""
    return clean_data(raw_dataframe, mock_config)


# =====================================================================
# >>> PROPUESTA 2: TESTS GRANULARES POR FUNCIONALIDAD
# =====================================================================

class TestColumnOperations:
    """Tests relacionados con operaciones de columnas."""
    
    def test_columns_renamed_correctly(self, clean_dataframe, mock_config):
        """Verifica que todas las columnas se renombraron según config."""
        expected_cols = ['col_num', 'col_str', 'col_outlier', 'col_rare', 'target']
        assert sorted(clean_dataframe.columns) == sorted(expected_cols)
    
    def test_drop_columns_removed(self, clean_dataframe):
        """Verifica que las columnas marcadas para drop fueron eliminadas."""
        assert 'col_drop' not in clean_dataframe.columns
        assert 'F_drop' not in clean_dataframe.columns
    
    def test_no_unexpected_columns(self, clean_dataframe, mock_config):
        """Verifica que no hay columnas adicionales no esperadas."""
        expected = set(['col_num', 'col_str', 'col_outlier', 'col_rare', 'target'])
        actual = set(clean_dataframe.columns)
        assert actual == expected


class TestNumericConversion:
    """Tests relacionados con conversión de tipos numéricos."""
    
    def test_numeric_columns_are_numeric_type(self, clean_dataframe):
        """Verifica que las columnas numéricas tienen dtype numérico."""
        assert pd.api.types.is_numeric_dtype(clean_dataframe['col_num'])
        assert pd.api.types.is_numeric_dtype(clean_dataframe['col_outlier'])
    
    def test_invalid_numeric_converted_to_nan(self, clean_dataframe):
        """Verifica que '?' se convierte en NaN."""
        # Posición 2 (índice 0-based después de filtrar) debería ser NaN
        assert clean_dataframe['col_num'].isna().sum() == 1
    
    def test_valid_numeric_strings_converted(self, clean_dataframe):
        """Verifica que strings numéricos válidos se convierten correctamente."""
        # '1', '2', '4', '5', '6', '7', '8', '9' → [1.0, 2.0, 5.0, 6.0, 7.0, 8.0, 9.0] + 1 NaN
        valid_values = clean_dataframe['col_num'].dropna()
        assert len(valid_values) == 7
        assert set(valid_values) == {1.0, 2.0, 5.0, 6.0, 7.0, 8.0, 9.0}


class TestOutlierHandling:
    """Tests relacionados con manejo de outliers."""
    
    def test_outliers_capped_to_nan(self, clean_dataframe):
        """Verifica que valores > 100 se convierten en NaN."""
        # Valores originales: 150 y 110 son > 100
        assert clean_dataframe['col_outlier'].isna().sum() == 2
    
    def test_valid_values_preserved(self, clean_dataframe):
        """Verifica que valores dentro del rango se preservan."""
        valid_values = clean_dataframe['col_outlier'].dropna()
        assert all(valid_values <= 100)
    
    def test_specific_outlier_values(self, clean_dataframe):
        """Verifica valores específicos después del capping."""
        expected = pd.Series([50.0, np.nan, 75.0, 90.0, 95.0, np.nan, 80.0, 85.0], name='col_outlier')
        pd_testing.assert_series_equal(clean_dataframe['col_outlier'], expected, check_index=False)


class TestStringCleaning:
    """Tests relacionados con limpieza de strings."""
    
    def test_whitespace_stripped(self, clean_dataframe):
        """Verifica que los espacios al inicio/final se eliminan."""
        # ' A ' se limpia a 'A', pero luego se elimina por ser rara
        # Verificamos que no hay strings con espacios
        str_values = clean_dataframe['col_str'].dropna()
        for val in str_values:
            assert val == val.strip()
    
    def test_garbage_strings_replaced(self, clean_dataframe):
        """Verifica que 'NaN' (string) se convierte en NaN real."""
        # Debe haber al menos 1 NaN del 'NaN' original
        assert clean_dataframe['col_str'].isna().sum() >= 1
    
    def test_valid_strings_preserved(self, clean_dataframe):
        """Verifica que strings válidos se preservan."""
        # 'B' aparece 6 veces y debe preservarse
        assert (clean_dataframe['col_str'] == 'B').sum() == 6


class TestRareCategoryHandling:
    """Tests relacionados con manejo de categorías raras."""
    
    def test_rare_categories_replaced_with_nan(self, clean_dataframe):
        """Verifica que categorías con frecuencia < 7 se vuelven NaN."""
        # 'rare1' aparece 1 vez, debe convertirse en NaN
        assert clean_dataframe['col_rare'].isna().sum() >= 1
    
    def test_frequent_categories_preserved(self, clean_dataframe):
        """Verifica que categorías frecuentes se preservan."""
        # 'cat1' aparece 7 veces, debe preservarse
        assert (clean_dataframe['col_rare'] == 'cat1').sum() == 7
    
    def test_no_rare_categories_in_final_data(self, clean_dataframe):
        """Verifica que no quedan categorías raras en los datos finales."""
        counts = clean_dataframe['col_rare'].value_counts()
        assert all(counts >= 7), "Todas las categorías restantes deben tener frecuencia >= 7"


class TestTargetProcessing:
    """Tests relacionados con procesamiento del target."""
    
    def test_invalid_target_rows_removed(self, clean_dataframe):
        """Verifica que filas con target inválido fueron eliminadas."""
        # Original: 9 filas, después de eliminar 'error': 8 filas
        assert clean_dataframe.shape[0] == 8
    
    def test_target_is_binary(self, clean_dataframe):
        """Verifica que el target solo contiene 0 y 1."""
        assert set(clean_dataframe['target'].unique()) == {0, 1}
    
    def test_target_is_integer_type(self, clean_dataframe):
        """Verifica que el target es de tipo entero."""
        assert clean_dataframe['target'].dtype == int
    
    def test_target_inversion_logic(self, clean_dataframe):
        """Verifica que el target fue invertido (1→0, 0→1)."""
        # Original (filtrado): [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
        # Invertido: [0, 1, 0, 1, 0, 0, 1, 0]
        expected = pd.Series([0, 1, 0, 1, 0, 0, 1, 0], name='target', dtype=int)
        pd_testing.assert_series_equal(clean_dataframe['target'], expected, check_index=False)


class TestDataQuality:
    """Tests relacionados con calidad general de los datos."""
    
    def test_no_duplicate_rows(self, clean_dataframe):
        """Verifica que no hay filas duplicadas."""
        assert clean_dataframe.duplicated().sum() == 0
    
    def test_index_reset_correctly(self, clean_dataframe):
        """Verifica que el índice fue reseteado correctamente."""
        assert list(clean_dataframe.index) == list(range(len(clean_dataframe)))
    
    def test_no_infinite_values(self, clean_dataframe):
        """Verifica que no hay valores infinitos."""
        numeric_cols = clean_dataframe.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert not np.isinf(clean_dataframe[col]).any()
    
    def test_expected_shape(self, clean_dataframe):
        """Verifica las dimensiones finales del DataFrame."""
        assert clean_dataframe.shape == (8, 5)  # 8 filas, 5 columnas


# =====================================================================
# >>> PROPUESTA 3: TESTS PARAMETRIZADOS PARA CASOS EDGE
# =====================================================================

@pytest.mark.parametrize("input_val,expected", [
    ('1', 1.0),
    ('?', np.nan),
    ('', np.nan),
    ('invalid', np.nan),
    (123, 123.0),
])
def test_numeric_conversion_edge_cases(input_val, expected, mock_config):
    """Prueba conversión numérica con diferentes casos edge."""
    df = pd.DataFrame({
        'A': [input_val],
        'B': ['dummy'],
        'C': [50],
        'D': ['cat'],
        'E': [1],
        'F': ['x']
    })
    
    result = clean_data(df, mock_config)
    
    if pd.isna(expected):
        assert pd.isna(result['col_num'].iloc[0])
    else:
        assert result['col_num'].iloc[0] == expected


@pytest.mark.parametrize("outlier_val,should_be_nan", [
    (50, False),
    (100, False),
    (101, True),
    (150, True),
    (9999, True),
])
def test_outlier_detection(outlier_val, should_be_nan, mock_config):
    """Prueba detección de outliers con diferentes umbrales."""
    df = pd.DataFrame({
        'A': ['1'],
        'B': ['B'],
        'C': [outlier_val],
        'D': ['cat'],
        'E': [1],
        'F': ['x']
    })
    
    result = clean_data(df, mock_config)
    
    if should_be_nan:
        assert pd.isna(result['col_outlier'].iloc[0])
    else:
        assert result['col_outlier'].iloc[0] == outlier_val


@pytest.mark.parametrize("garbage_str", [
    'NaN', 'nan', 'NAN', 'null', 'NULL', 'Null',
    'na', 'NA', 'N/A', 'n/a', 'none', 'None', 'NONE',
    '  NaN  ', '  null  ', '  na  '
])
def test_garbage_string_detection(garbage_str, mock_config):
    """Prueba detección de diferentes variantes de strings basura."""
    df = pd.DataFrame({
        'A': ['1', '2', '3', '4', '5', '6', '7', '8'],
        'B': [garbage_str, 'B', 'B', 'B', 'B', 'B', 'B', 'B'],
        'C': [50, 60, 70, 80, 90, 95, 85, 75],
        'D': ['cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat'],
        'E': [1, 0, 1, 0, 1, 0, 1, 0],
        'F': ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x']
    })
    
    result = clean_data(df, mock_config)
    
    # El primer valor debería ser NaN
    assert pd.isna(result['col_str'].iloc[0]), f"'{garbage_str}' debería detectarse como basura"


# =====================================================================
# >>> PROPUESTA 4: TESTS DE INTEGRACIÓN
# =====================================================================

def test_full_pipeline_preserves_data_integrity(raw_dataframe, mock_config):
    """Prueba que el pipeline completo mantiene la integridad de los datos."""
    result = clean_data(raw_dataframe, mock_config)
    
    # Verificaciones generales
    assert result.shape[0] <= raw_dataframe.shape[0]  # No debe aumentar filas
    assert len(result.columns) <= len(raw_dataframe.columns)  # Algunas columnas se eliminan
    assert result['target'].notna().all()  # Target no debe tener NaN


def test_pipeline_is_deterministic(raw_dataframe, mock_config):
    result_first = clean_data(raw_dataframe.copy(), mock_config)
    result_second = clean_data(raw_dataframe.copy(), mock_config)
    pd_testing.assert_frame_equal(result_first, result_second)  # Los resultados deben ser iguales

# =====================================================================
# >>> PROPUESTA 5: TESTS DE VALIDACIÓN DE CONFIG
# =====================================================================

def test_clean_data_handles_missing_config_keys(raw_dataframe):
    """Verifica manejo de configuraciones incompletas."""
    incomplete_config = {
        'data_cleaning': {
            'rename_cols': ['col_num', 'col_str', 'col_outlier', 'col_rare', 'target', 'col_drop'],
            'outlier_cols': {}  # Sin outliers definidos
        },
        'base': {'target_col': 'target'},
        'preprocessing': {
            'numeric': {'features': ['col_num', 'col_outlier']},
            'nominal': {'features': ['col_str']},
            'ordinal': {'features': ['col_rare']}
        }
    }
    
    # No debería lanzar error
    result = clean_data(raw_dataframe, incomplete_config)
    assert result is not None


# =====================================================================
# >>> PROPUESTA 6: TESTS DE PERFORMANCE (OPCIONAL)
# =====================================================================
# @pytest.mark.skipif(
#     not hasattr(pytest, 'benchmark'),
#     reason="pytest-benchmark no está instalado"
# )
# def test_clean_data_performance(raw_dataframe, mock_config, benchmark):
#     """Prueba de performance usando pytest-benchmark."""
#     # Requiere: pip install pytest-benchmark
#     # Uso: pytest --benchmark-only
#     result = benchmark(clean_data, raw_dataframe, mock_config)
#     assert result is not None