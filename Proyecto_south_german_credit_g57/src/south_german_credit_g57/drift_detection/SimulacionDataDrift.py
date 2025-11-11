# ================================================
# SCRIPT: 01_generate_drift_auto.py
# Simula Data Drift automáticamente detectando tipos
# ================================================

import pandas as pd
import numpy as np
import yaml
import os
import argparse
from pathlib import Path
import random


def load_config(config_path: str) -> dict:
    """Carga configuración YAML con tolerancia a .yml/.yaml."""
    if not os.path.exists(config_path):
        alt_path = config_path.replace(".yaml", ".yml") if config_path.endswith(".yaml") else config_path.replace(".yml", ".yaml")
        if os.path.exists(alt_path):
            config_path = alt_path
        else:
            raise FileNotFoundError(f"No se encontró el archivo YAML: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def generar_data_drift_auto(input_csv, output_csv, intensidad_global=0.2, seed=42):
    """Aplica drift automático a columnas numéricas y categóricas."""
    np.random.seed(seed)
    random.seed(seed)

    df = pd.read_csv(input_csv)
    print(f"\n📂 Archivo cargado: {input_csv} ({df.shape[0]} filas, {df.shape[1]} columnas)")

    # Detectar columnas numéricas y categóricas
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    print(f"🔢 Columnas numéricas detectadas: {len(numeric_cols)}")
    print(f"🔠 Columnas categóricas detectadas: {len(cat_cols)}")

    columnas_modificadas = []

    # --- Aplicar drift numérico ---
    for col in numeric_cols:
        direccion = random.choice(["subir", "bajar", "ambos"])
        std = df[col].std()
        ruido = np.random.normal(0, intensidad_global * std, df.shape[0])

        if direccion == "subir":
            df[col] += abs(ruido)
        elif direccion == "bajar":
            df[col] -= abs(ruido)
        else:
            df[col] += ruido

        columnas_modificadas.append((col, "numérico", direccion))
        print(f"   ✅ {col}: drift numérico ({direccion}, intensidad={intensidad_global})")

    # --- Aplicar drift categórico ---
    for col in cat_cols:
        if df[col].nunique() <= 1:
            continue
        valores = df[col].unique()
        p = np.random.dirichlet(np.ones(len(valores)))
        df[col] = np.random.choice(valores, size=len(df), p=p)
        columnas_modificadas.append((col, "categórico", "reajuste distribución"))
        print(f"   🔁 {col}: drift categórico (reajuste de distribución)")

    # Guardar resultado
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Dataset con Data Drift guardado en: {output_csv}")

    # Guardar log de cambios
    log_path = Path(output_csv).with_suffix(".log")
    with open(log_path, "w") as f:
        for col, tipo, direccion in columnas_modificadas:
            f.write(f"{col}: {tipo} - {direccion}\n")
    print(f"🧾 Log de drift guardado en: {log_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Simula Data Drift automático detectando columnas.")
    parser.add_argument("--config", required=True, help="Ruta del archivo YAML de configuración.")
    args = parser.parse_args()

    # Cargar configuración
    config = load_config(args.config)

    # Detectar raíz del proyecto automáticamente (sube dos niveles desde drift_detection)
    root_dir = Path(__file__).resolve().parents[3]

    # Construir rutas correctas
    input_path = (root_dir / config["data"]["input"]).resolve()
    output_path = (root_dir / config["data"]["output"]).resolve()

    print(f"\n📍 Proyecto raíz detectado: {root_dir}")
    print(f"📂 Archivo de entrada detectado: {input_path}")
    print(f"📁 Archivo de salida será: {output_path}")

    # Leer configuración de drift
    drift_conf = config.get("drift", {})
    intensidad = drift_conf.get("intensidad_global", 0.2)
    seed = drift_conf.get("seed", 42)

    generar_data_drift_auto(input_path, output_path, intensidad_global=intensidad, seed=seed)


if __name__ == "__main__":
    main()
