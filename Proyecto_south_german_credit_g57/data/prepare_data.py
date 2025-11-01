# ============================================
# 📦 Módulo: prepare_data.py
# --------------------------------------------
# Objetivo:
# - Cargar y preparar el dataset de crédito
# - Convertir tipos numéricos
# - Invertir la variable objetivo (credit_risk)
# - Dividir datos en entrenamiento y prueba
# ============================================

import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(path: str) -> pd.DataFrame:
    """
    Carga el dataset desde un archivo CSV o Parquet.
    """
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        raise ValueError("❌ Formato no soportado. Use .csv o .parquet")

    print(f"📂 Dataset cargado desde: {path}")
    print(f"   Dimensiones: {df.shape}")
    return df


def prepare_dataset(
    df: pd.DataFrame,
    target_col: str = "credit_risk",
    num_cols: list = None,
    cat_cols: list = None,
    ord_cols: list = None,
    test_size: float = 0.30,
    random_state: int = 57
):
    """
    Prepara el dataset para modelado:
    - Convierte valores a numéricos
    - Invierte el target (1→0, 0→1)
    - Realiza split estratificado train/test
    """

    df = df.copy()

    # ===== Conversión global de tipos numéricos =====
    # (esto es lo que tú hacías con df.apply(pd.to_numeric))
    df = df.apply(pd.to_numeric, errors="ignore")

    # =====  Asegurar tipo int para variables categóricas numéricas =====
    df = df.astype("int64", errors="ignore")

    # =====  Inversión de la variable objetivo =====
    if target_col in df.columns:
        df[target_col] = df[target_col].apply(lambda x: 0 if x == 1 else 1)
        print(f"\n🔄 Se invirtieron los valores de '{target_col}' (1→0, 0→1)")
        print(df[target_col].value_counts())

    # =====  Separación de variables =====
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # ===== División reproducible =====
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # ===== Resumen de partición =====
    print("\nDivisión del dataset:")
    print(f"   Entrenamiento: {Xtrain.shape} | Prueba: {Xtest.shape}")

    pct_pos = (ytrain.sum() / ytrain.shape[0]) * 100
    pct_neg = 100 - pct_pos
    print(f"   Balance clases (train): Positiva {pct_pos:.2f}% | Negativa {pct_neg:.2f}%")

    return Xtrain, Xtest, ytrain, ytest
