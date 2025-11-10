import pytest
import numpy as np
import warnings

from south_german_credit_g57.evaluation.metrics_module import calculate_classification_metrics

# =====================================================================
# >>> FIXTURES CON CASOS DE PRUEBA
# =====================================================================

@pytest.fixture(params=[
    ("perfect", {
        'y_true': np.array([0, 1, 0, 1]),
        'y_pred': np.array([0, 1, 0, 1]),
        'y_prob': np.array([0.1, 0.9, 0.2, 0.8]),
        'expected': {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'roc_auc': 1.0}
    }),
    ("awful", {
        'y_true': np.array([0, 1, 0, 1]),
        'y_pred': np.array([1, 0, 1, 0]),
        'y_prob': np.array([0.9, 0.1, 0.8, 0.2]),
        'expected': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'roc_auc': 0.0}
    }),
    ("real_mixed", {
        # y_true = [0, 1, 0, 1, 0, 0] (2 Pos, 4 Neg)
        # y_pred = [0, 1, 1, 0, 0, 0] (2 Pos Pred, 4 Neg Pred)
        # TP=1, FP=1, TN=3, FN=1
        'y_true': np.array([0, 1, 0, 1, 0, 0]),
        'y_pred': np.array([0, 1, 1, 0, 0, 0]),
        'y_prob': np.array([0.1, 0.9, 0.7, 0.3, 0.2, 0.4]),
        'expected': {
            'accuracy': (1+3)/6,  # 0.666...
            'precision': 1/(1+1), # 0.5
            'recall': 1/(1+1),    # 0.5
            'f1': 0.5,
            'roc_auc': 0.75
        }
    }),
    ("imbalanced_null_classifier", {
        # Clasificador que siempre predice 0 en data desbalanceada
        # y_true = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1] (1 Pos, 9 Neg)
        # y_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # TP=0, FP=0, TN=9, FN=1
        'y_true': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
        'y_pred': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'y_prob': np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2]),
        'expected': {
            'accuracy': 0.9,
            'precision': 0.0, # Genera ZeroDivisionWarning
            'recall': 0.0,
            'f1': 0.0,
            'roc_auc': 1.0
        }
    })
], ids=["perfect_score", "awful_score", "real_mixed_score", "imbalanced_null_score"])
def metrics_test_case(request):
    """Fixture parametrizada que provee todos los casos de prueba."""
    return request.param

# =====================================================================
# >>> CLASE DE PRUEBA: TestMetricsCalculation
# =====================================================================

class TestMetricsCalculation:
    """Pruebas para el cálculo de métricas (calculate_classification_metrics)."""

    def test_all_metrics_calculation(self, metrics_test_case):
        """Verifica el cálculo de todas las métricas usando parametrize."""
        name, data = metrics_test_case
        
        # Suprimimos el ZeroDivisionWarning esperado en el caso "imbalanced"
        # (precision = 0 / 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning) # sklearn usa UserWarning
            warnings.simplefilter("ignore", category=RuntimeWarning) # a veces RuntimeWarning
            
            result_metrics = calculate_classification_metrics(
                data['y_true'], 
                data['y_pred'], 
                data['y_prob']
            )

        expected_metrics = data['expected']
        
        # Usamos pytest.approx() para comparar floats de forma segura
        assert result_metrics['accuracy'] == pytest.approx(expected_metrics['accuracy'])
        assert result_metrics['precision'] == pytest.approx(expected_metrics['precision'])
        assert result_metrics['recall'] == pytest.approx(expected_metrics['recall'])
        assert result_metrics['f1'] == pytest.approx(expected_metrics['f1'])
        assert result_metrics['roc_auc'] == pytest.approx(expected_metrics['roc_auc'])

    def test_roc_auc_handles_none_prob(self):
        """Verifica que ROC-AUC es None o NaN si y_prob no se provee."""
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 1, 1])
        
        result_metrics = calculate_classification_metrics(y_true, y_pred, y_prob=None)
        
        # Verifica que las otras métricas sí existen
        assert 'accuracy' in result_metrics
        assert 'precision' in result_metrics
        
        # Verifica que roc_auc es None o NaN
        assert 'roc_auc' not in result_metrics or \
            result_metrics['roc_auc'] is None or \
            np.isnan(result_metrics['roc_auc'])

    def test_confusion_matrix_calculation(self):
        """Verifica el cálculo de la matriz de confusión."""
        # Basado en el caso "real_mixed"
        y_true = np.array([0, 1, 0, 1, 0, 0])
        y_pred = np.array([0, 1, 1, 0, 0, 0])
        
        # sklearn.metrics.confusion_matrix(y_true, y_pred)
        # TN=3, FP=1
        # FN=1, TP=1
        expected_cm = np.array([[3, 1], [1, 1]])

        result_metrics = calculate_classification_metrics(y_true, y_pred, y_prob=None)
        
        assert 'confusion_matrix' in result_metrics
        assert np.array_equal(result_metrics['confusion_matrix'], expected_cm)