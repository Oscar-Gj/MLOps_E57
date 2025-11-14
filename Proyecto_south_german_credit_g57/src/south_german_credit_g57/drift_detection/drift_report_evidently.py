# ================================================
# SCRIPT: 03_drift_report_evidently.py
# Genera un reporte visual de Data Drift con Evidently
# ================================================

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping
from pathlib import Path
import argparse
import os

def generar_reporte_drift(reference_path, current_path, target_col, output_dir="reports/drift"):
    """Genera un reporte de Data Drift usando Evidently."""
    print("\n📊 Generando reporte de Data Drift...")

    ref = pd.read_csv(reference_path)
    cur = pd.read_csv(current_path)

    # Definir mapeo de columnas
    column_mapping = ColumnMapping()
    column_mapping.target = target_col

    # Crear reporte Evidently
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur, column_mapping=column_mapping)

    # Crear carpeta de salida con fecha
    output_dir = Path(output_dir) / pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M")
    os.makedirs(output_dir, exist_ok=True)

    html_path = output_dir / "data_drift_report.html"
    json_path = output_dir / "data_drift_summary.json"

    report.save_html(str(html_path))
    report.save_json(str(json_path))


    print(f"✅ Reporte HTML generado: {html_path}")
    print(f"📁 Resumen JSON guardado: {json_path}")
    return html_path, json_path


def main():
    parser = argparse.ArgumentParser(description="Genera un reporte de Data Drift con Evidently.")
    parser.add_argument("--ref", required=True, help="Ruta al dataset de referencia (original test).")
    parser.add_argument("--cur", required=True, help="Ruta al dataset con drift.")
    parser.add_argument("--target", required=True, help="Nombre de la variable objetivo.")
    parser.add_argument("--output", default="reports/drift", help="Carpeta de salida del reporte.")
    args = parser.parse_args()

    generar_reporte_drift(args.ref, args.cur, args.target, args.output)


if __name__ == "__main__":
    main()
