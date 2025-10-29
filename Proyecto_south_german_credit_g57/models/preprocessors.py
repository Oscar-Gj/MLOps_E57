# ============================================
# Módulo: preprocessors.py
# --------------------------------------------
# Objetivo:
# - Definir los pipelines de preprocesamiento
# - Crear transformadores personalizados (BinaryEncoderWrapper)
# - Construir el ColumnTransformer final para el pipeline
# ============================================

import pandas as pd
import numpy as np
import category_encoders as ce

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


# =====  Clase personalizada: BinaryEncoderWrapper =====
class BinaryEncoderWrapper(BaseEstimator, TransformerMixin):
    """
    Envoltorio para aplicar Binary Encoding de category_encoders
    dentro de un pipeline de Scikit-Learn.
    """

    def __init__(self, cols=None):
        self.cols = cols
        self.encoder = ce.BinaryEncoder(cols=self.cols)
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        X_df = self._to_df(X)
        self.encoder.fit(X_df)
        # Guardamos nombres de las columnas generadas
        try:
            self.feature_names_out_ = list(self.encoder.get_feature_names_out())
        except Exception:
            self.feature_names_out_ = list(self.encoder.transform(X_df).columns)
        return self

    def transform(self, X):
        X_df = self._to_df(X)
        out = self.encoder.transform(X_df)
        return pd.DataFrame(out, columns=out.columns, index=X_df.index)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_ or [], dtype=object)

    def _to_df(self, X):
        """Convierte arrays o listas en DataFrame."""
        if isinstance(X, pd.DataFrame):
            return X
        cols = self.cols if self.cols is not None else [f"col_{i}" for i in range(np.asarray(X).shape[1])]
        return pd.DataFrame(X, columns=cols)


# =====  Función para construir el preprocesador =====
def build_preprocessor(num_cols, cat_cols, ord_cols):
    """
    Crea un ColumnTransformer con tres sub-pipelines:
    - numéricos: imputación, escalado, transformación
    - categóricos: imputación, codificación binaria
    - ordinales: imputación, codificación ordinal
    """

    # --- Numéricos ---
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler(feature_range=(0, 1))),
        ("power", PowerTransformer(method="yeo-johnson")),
    ])

    # --- Categóricos nominales ---
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("binary", BinaryEncoderWrapper(cols=cat_cols))
    ])

    # --- Categóricos ordinales ---
    ord_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder())
    ])

    # --- ColumnTransformer general ---
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
            ("ord", ord_pipe, ord_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    print(" Preprocesador construido correctamente")
    print(f"   Numéricas: {len(num_cols)} | Categóricas: {len(cat_cols)} | Ordinales: {len(ord_cols)}")

    return preprocessor
