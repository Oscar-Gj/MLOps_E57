import mlflow
import random
import numpy as np
import pandas as pd
from typing import Any


class RandomPredictor(mlflow.pyfunc.PythonModel):
    """
    Predicción de Prueba.

    Args:
        context: Prueba de MLFlow.
        model_input: Los datos de entradas son para pruebas y sintéticos.

    Returns:
        Una predicción sencilla al azar.
    """

    def predict(self, context: Any, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        predictions = [self.random_prediction(model_input)]
        return pd.DataFrame(predictions)

    @staticmethod
    def random_prediction(params=None):
        return np.array(random.randint(0, 1), dtype=np.float32)