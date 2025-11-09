import pandas as pd
import numpy as np
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict
from sklearn.model_selection import train_test_split

# ================================
# CONFIGURACIÓN GLOBAL Y LOGGER
# ================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Rutas base dinámicas ===
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]  # Subir dos niveles desde /src/south_german_credit_g57/
DATA_DIR = PROJECT_ROOT / "data"

def resolve_path(relative_path: str) -> Path:
    """Convierte rutas relativas a absolutas desde la raíz del proyecto."""
    path = Path(relative_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


# ================================
# FUNCIONES PRINCIPALES
# ================================
def load_config(config_path: str) -> Dict:
    """Carga la configuración YAML en UTF-8."""
    logger.info(f"Cargando configuración desde: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_data(path: str) -> pd.DataFrame:
    """Carga los datos desde CSV o Parquet."""
    full_path = resolve_path(path)
    logger.info(f"Cargando datos desde: {full_path}")
    if full_path.suffix == ".csv":
        return pd.read_csv(full_path)
    elif full_path.suffix == ".parquet":
        return pd.read_parquet(full_path)
    else:
        raise ValueError(f"Formato no soportado: {full_path.suffix}")


def clean_data(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Limpieza completa del dataset según la configuración."""
    logger.info("Iniciando limpieza de datos...")
    df_clean = df.copy()

    # Renombrar columnas
    rename_cols = config['data_cleaning']['rename_cols']
    df_clean.columns = rename_cols
    logger.info("Columnas renombradas correctamente.")

    # Eliminar columnas innecesarias
    drop_cols = config['data_cleaning'].get('drop_cols', [])
    if drop_cols:
        df_clean = df_clean.drop(columns=drop_cols)
        logger.info(f"Columnas eliminadas: {drop_cols}")

    target = config['base']['target_col']
    num_cols = config['preprocessing']['numeric']['features']
    obj_cols = config['preprocessing']['nominal']['features'] + config['preprocessing']['ordinal']['features']

    # Reemplazar strings basura
    # garbage_strings = ['?', 'null', 'invalid', 'error', ' NAN ', ' INVALID ', ' ERROR ', ' n/a ']
    garbage_strings = [
        '', 'na', 'n/a', 'none', 'null', 'nil', 'nan',
        'missing', 'unknown', 'error', 'invalid', 'undefined', 'unavailable'
    ]
    # df_clean = df_clean.replace(garbage_strings, np.nan)

    # Limpiar espacios
    for col in obj_cols:
        if col in df_clean.columns:
            # 1. Convertir a string y quitar espacios al inicio/final
            df_clean[col] = df_clean[col].astype(str).str.strip()
            
            # 2. Crear una versión normalizada (minúsculas) para comparación
            normalized = df_clean[col].str.lower()
            
            # 3. Reemplazar valores cuya versión normalizada está en garbage_strings
            mask = normalized.isin(garbage_strings)
            df_clean.loc[mask, col] = np.nan

    # Convertir columnas numéricas
    for col in num_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # Corregir outliers
    outlier_caps = config['data_cleaning']['outlier_cols']
    for col, cap in outlier_caps.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].mask(df_clean[col] > cap, pd.NA)

    # Reemplazar categorías raras
    for col in obj_cols:
        if col in df_clean.columns:
            counts = df_clean[col].value_counts()
            rare_categories = counts[counts < 7].index
            if len(rare_categories) > 0:
                mask = df_clean[col].isin(rare_categories)
                df_clean[col] = df_clean[col].mask(mask, np.nan)
                df_clean[col] = df_clean[col].astype(object)

    # Target
    df_clean[target] = pd.to_numeric(df_clean[target], errors='coerce')
    before = df_clean.shape[0]
    df_clean = df_clean[df_clean[target].isin([0.0, 1.0])]
    logger.info(f"Filas eliminadas por target inválido: {before - df_clean.shape[0]}")

    df_clean[target] = df_clean[target].apply(lambda x: 0 if x == 1 else 1).astype(int)
    df_clean = df_clean.reset_index(drop=True)
    logger.info(f"Limpieza finalizada. Shape: {df_clean.shape}")

    return df_clean


def save_data(df: pd.DataFrame, config: Dict):
    """Guarda los datasets procesados y los divide en train/test."""
    processed_path = resolve_path(config['data']['processed'])
    train_path = resolve_path(config['data']['train'])
    test_path = resolve_path(config['data']['test'])
    target = config['base']['target_col']
    split_params = config['data_cleaning']

    # Crear directorio de salida
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    # Guardar completo
    logger.info(f"Guardando datos procesados en: {processed_path}")
    df.to_csv(processed_path, index=False)

    # Split Train/Test
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=split_params['test_size'],
        random_state=split_params['random_state'],
        stratify=y
    )

    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    # Guardar train/test
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Guardando datos de entrenamiento en: {train_path}")
    df_train.to_csv(train_path, index=False)

    logger.info(f"Guardando datos de prueba en: {test_path}")
    df_test.to_csv(test_path, index=False)


# ================================
# ENTRYPOINT
# ================================
def main(config_path: str):
    """Orquestador principal para carga, limpieza y guardado."""
    try:
        config = load_config(config_path)
        df_raw = load_data(config['data']['raw'])
        df_clean = clean_data(df_raw, config)
        save_data(df_clean, config)
        logger.info("Proceso de limpieza completado exitosamente.")
    except Exception as e:
        logger.error(f"Error en el proceso: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de limpieza de datos para el proyecto Credit Risk.")
    parser.add_argument('--config', type=str, default='params.yaml', help='Ruta al archivo YAML de configuración.')
    args = parser.parse_args()
    main(config_path=args.config)
