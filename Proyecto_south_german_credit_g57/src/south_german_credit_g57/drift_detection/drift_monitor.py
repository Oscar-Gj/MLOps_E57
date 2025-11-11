# ==========================================================
# SCRIPT: 04_drift_monitor.py
# Detección global y por columna de Data Drift + registro
# ==========================================================

import json
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def obtener_reporte_mas_reciente(base_dir="reports/drift"):
    """Busca el último subdirectorio con un reporte Evidently."""
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"No existe la carpeta de reportes: {base_path}")

    subdirs = [d for d in base_path.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No se encontraron subcarpetas en {base_path}")

    subdirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_dir = subdirs[0]

    json_path = latest_dir / "data_drift_summary.json"
    if not json_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo JSON en {latest_dir}")

    print(f"📂 Último reporte detectado: {latest_dir.name}")
    return json_path


def analizar_drift_columnas(json_path, top_n=5):
    """
    Analiza drift columna por columna a partir del JSON de Evidently.
    Retorna un DataFrame con las columnas afectadas.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    metrics = data["metrics"][0]["result"]
    columns_info = metrics.get("drift_by_columns", {})

    if not columns_info:
        print("⚠️ No hay información detallada de columnas en el JSON.")
        return pd.DataFrame()

    registros = []
    for col, info in columns_info.items():
        registros.append({
            "columna": col,
            "drift_detectado": info.get("drift_detected", False),
            "p_value": info.get("p_value"),
            "stat_test": info.get("stattest_name", "N/A")
        })

    df = pd.DataFrame(registros)
    drifted = df[df["drift_detectado"] == True]

    print(f"\n📊 Análisis por columna:")
    print(f"   Total de columnas: {len(df)}")
    print(f"   Columnas con drift: {len(drifted)} ({len(drifted)/len(df):.1%})")

    if not drifted.empty:
        print("\n🔍 Columnas más afectadas:")
        print(drifted.sort_values("p_value").head(top_n).to_string(index=False))
    else:
        print("\n✅ Ninguna columna presenta drift significativo.")

    return drifted


def analizar_drift(json_path, threshold_drift_rate=0.2):
    """Evalúa el nivel de drift y propone acción."""
    with open(json_path, "r") as f:
        data = json.load(f)

    summary = data["metrics"][0]["result"]
    drift_share = summary.get("share_of_drifted_columns", 0)
    total_cols = summary.get("number_of_columns", 0)
    drifted_cols = summary.get("number_of_drifted_columns", 0)

    print("\n📊 Resultados globales del análisis de Drift:")
    print(f"   Columnas totales: {total_cols}")
    print(f"   Columnas con drift: {drifted_cols}")
    print(f"   Proporción de drift: {drift_share:.2%}")

    # Clasificación de severidad
    if drift_share == 0:
        nivel = "🟢 Sin drift detectable"
        accion = "No se requiere acción."
    elif drift_share < threshold_drift_rate:
        nivel = "🟡 Drift leve"
        accion = "Monitorear próximas ejecuciones."
    elif drift_share < 0.5:
        nivel = "🟠 Drift moderado"
        accion = "Revisar pipeline de features o data sources recientes."
    else:
        nivel = "🔴 Drift severo"
        accion = "⚙️ Reentrenar modelo."

    print(f"\n📈 Nivel de Drift: {nivel}")
    print(f"💡 Acción recomendada: {accion}")

    # Registrar resultado global
    registro = {
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_columns": total_cols,
        "drifted_columns": drifted_cols,
        "drift_share": drift_share,
        "nivel": nivel,
        "accion": accion,
        "source": str(json_path),
    }

    return registro, nivel, accion


def main():
    base_dir = Path(__file__).resolve().parent / "reports/drift"
    json_path = obtener_reporte_mas_reciente(base_dir)

    registro, nivel, accion = analizar_drift(json_path, threshold_drift_rate=0.2)
    drifted_cols_df = analizar_drift_columnas(json_path, top_n=8)

    # Guardar log consolidado
    logs_dir = Path(json_path).parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / "drift_monitor_log.csv"

    # Agregar top columnas (solo nombres)
    if not drifted_cols_df.empty:
        top_cols = ", ".join(drifted_cols_df["columna"].head(5).tolist())
    else:
        top_cols = "Ninguna"

    registro["top_columns_drift"] = top_cols

    df = pd.DataFrame([registro])
    if log_file.exists():
        df.to_csv(log_file, mode="a", header=False, index=False)
    else:
        df.to_csv(log_file, index=False)

    print(f"\n🧾 Registro actualizado: {log_file}")


if __name__ == "__main__":
    main()
