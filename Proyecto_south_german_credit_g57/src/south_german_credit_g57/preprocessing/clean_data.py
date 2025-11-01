import pandas as pd
import numpy as np
import yaml
import argparse
import logging
from typing import List, Dict, Any

# Configurar un logger simple
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict:
    """Carga la configuración desde un archivo YAML."""
    logger.info(f"Cargando configuración desde: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_data(path: str) -> pd.DataFrame:
    """Carga los datos crudos desde un CSV."""
    logger.info(f"Cargando datos crudos desde: {path}")
    return pd.read_csv(path)

def clean_data(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Aplica todas las reglas de limpieza, renombrado y corrección de outliers
    identificadas en el notebook de experimentación.
    """
    logger.info("Iniciando limpieza de datos...")
    df_clean = df.copy()

    # 1. Renombrar columnas (según celda [44] del notebook)
    logger.info("Renombrando columnas...")
    rename_cols = config['data_cleaning']['rename_cols']
    df_clean.columns = rename_cols

    # 2. Eliminar columnas innecesarias (según celda [45])
    drop_cols = config['data_cleaning']['drop_cols']
    df_clean = df_clean.drop(columns=drop_cols)
    logger.info(f"Columnas eliminadas: {drop_cols}")

    # 3. Definir listas de features y target
    target = config['base']['target_col']
    num_cols = config['features']['numeric']
    
    # Combinar features categóricas y ordinales para limpieza de texto
    obj_cols = config['features']['categorical'] + config['features']['ordinal']

    # 4. Reemplazar todos los strings de basura con NaN (identificados en celda [54])
    logger.info("Reemplazando valores de texto erróneos por NaN...")
    garbage_strings = ['?', 'null', 'invalid', 'error', ' NAN ', ' INVALID ', ' ERROR ', ' n/a ']
    df_clean = df_clean.replace(garbage_strings, np.nan)

    # 5. Limpiar espacios en blanco de columnas de texto (según celda [55])
    for col in obj_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip()

    # 6. Convertir columnas numéricas, forzando errores a NaN (según celda [47])
    for col in num_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # 7. Corregir/Tapar outliers, convirtiéndolos en NaN (según celda [52])
    logger.info("Corrigiendo outliers...")
    outlier_caps = config['data_cleaning']['outlier_caps']
    for col, cap in outlier_caps.items():
        if col in df_clean.columns:
            # Usamos pd.NA para compatibilidad con el tipo Float64
            df_clean[col] = df_clean[col].mask(df_clean[col] > cap, pd.NA)

    # 8. Reemplazar categorías raras por NaN (según celda [55])
    logger.info("Reemplazando categorías raras por NaN...")
    for col in obj_cols:
        if col in df_clean.columns:
            counts = df_clean[col].value_counts()
            # Frecuencia mínima definida en el notebook (celda [55] usa < 7)
            rare_categories = counts[counts < 7].index
            df_clean[col] = df_clean[col].replace(rare_categories, np.nan)

    # 9. Limpiar variable objetivo (TARGET) (según celda [55] y [70])
    logger.info(f"Limpiando y transformando la columna objetivo: {target}")
    
    # Convertir a numérico, forzando errores (como 'invalid') a NaN
    df_clean[target] = pd.to_numeric(df_clean[target], errors='coerce')
    
    # Quedarse solo con filas que tienen un target válido (0 o 1)
    original_rows = df_clean.shape[0]
    df_clean = df_clean[df_clean[target].isin([0.0, 1.0])]
    logger.info(f"Filas eliminadas por target inválido: {original_rows - df_clean.shape[0]}")
    
    # Invertir el target: 1 (bueno) -> 0, 0 (malo) -> 1 (según celda [70])
    df_clean[target] = df_clean[target].apply(lambda x: 0 if x == 1 else 1)
    
    # Asegurar que el target sea entero
    df_clean[target] = df_clean[target].astype(int)

    # 10. Resetear el índice (según celda [67])
    df_clean = df_clean.reset_index(drop=True)
    
    logger.info(f"Limpieza finalizada. Shape final: {df_clean.shape}")
    return df_clean

def save_data(df: pd.DataFrame, path: str):
    """Guarda el DataFrame limpio en la carpeta de procesados."""
    logger.info(f"Guardando datos procesados en: {path}")
    df.to_csv(path, index=False)

def main(config_path: str):
    """Orquestador principal para cargar, limpiar y guardar los datos."""
    try:
        config = load_config(config_path)
        
        df_raw = load_data(config['data']['raw'])
        df_clean = clean_data(df_raw, config)
        save_data(df_clean, config['data']['processed'])
        
        logger.info("Proceso de limpieza completado exitosamente.")
        
    except FileNotFoundError as e:
        logger.error(f"Error: Archivo no encontrado. {e}")
    except Exception as e:
        logger.error(f"Ocurrió un error en el proceso: {e}")

if __name__ == "__main__":
    # Configurar argparse para aceptar la ruta del config
    parser = argparse.ArgumentParser(description="Script de limpieza de datos para el proyecto de riesgo crediticio.")
    parser.add_argument('--config', type=str, default='params.yaml', help='Ruta al archivo de configuración YAML.')
    
    # Parsear los argumentos
    args = parser.parse_args()
    main(config_path=args.config)