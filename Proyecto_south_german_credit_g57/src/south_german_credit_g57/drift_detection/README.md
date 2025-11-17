
# Módulo de Monitoreo y Simulación de Data Drift

Este módulo forma parte del proyecto `Proyecto_south_german_credit_g57` y tiene como objetivo simular, detectar y monitorear cambios estadísticos (*data drift*) entre datasets de entrenamiento, prueba o producción.

---

## Estructura del módulo

drift_detection/
│
├── SimulacionDataDrift.py
├── drift_report_evidently.py
├── drift_monitor.py
├── params_drift.yml
├── reports/
│ └── drift/
│ └── YYYY-MM-DD_HH-MM/
│ ├── data_drift_report.html
│ ├── data_drift_summary.json
│ └── logs/
│ └── drift_monitor_log.csv
└── README.md

---

## 1. Archivo de configuración (`params_drift.yml`)

```yaml
data:
  input: "data/processed/02_df_data_test_01.csv"
  output: "data/processed/02_df_data_drift_auto.csv"

drift:
  seed: 42
  intensidad_global: 0.25
  target_col: "credit_risk"
  exclude_cols: ["id"]
2. Simulación de Data Drift (dataset artificial)

Genera un dataset con drift controlado aplicando ruido numérico y redistribución categórica.

Comando:
python 01_generate_drift_auto.py --config params_drift.yml

Resultado:

Dataset modificado → data/processed/02_df_data_drift_auto.csv

Log de columnas afectadas → data/processed/02_df_data_drift_auto.log

3. Generar reporte con EvidentlyAI

Compara el dataset original y el dataset con drift usando EvidentlyAI.

Comando:
python drift_report_evidently.py \
  --ref ../../../data/processed/02_df_data_test_01.csv \
  --cur ../../../data/processed/02_df_data_drift_auto.csv \
  --target credit_risk
Resultado:

Se genera dentro de reports/drift/:

reports/drift/2025-11-14_22-10/
├── data_drift_report.html
├── data_drift_summary.json
└── logs/
    └── drift_monitor_log.csv


El HTML incluye distribución antes/después, p-values, tests estadísticos y visualizaciones clave.

4. Monitoreo del último reporte generado

Evalúa el nivel de drift automáticamente leyendo el reporte más reciente.

Comando:
python 04_drift_monitor.py

Resultado:

Nivel de drift 

% de columnas afectadas

Columnas más afectadas

Acción recomendada

Log actualizado en:

reports/drift/YYYY-MM-DD_HH-MM/logs/drift_monitor_log.csv

5. Flujo completo (ejecutar en orden)
python 01_generate_drift_auto.py --config params_drift.yml

python drift_report_evidently.py \
  --ref ../../../data/processed/02_df_data_test_01.csv \
  --cur ../../../data/processed/02_df_data_drift_auto.csv \
  --target credit_risk

python 04_drift_monitor.py

Interpretación del nivel de drift
Nivel	Umbral	Significado	Acción
 Sin drift	0%	Todo estable	Ninguna
 Leve	< 20%	Cambios ligeros	Monitorear
 Moderado	20–50%	Cambios significativos	Revisar pipeline / datos
 Severo	> 50%	Cambios fuertes	Reentrenar modelo

Archivos generados
Archivo	Descripción
*_drift_auto.csv	Dataset con drift
*.log	Columnas afectadas
data_drift_report.html	Reporte visual Evidently
data_drift_summary.json	Resultados estadísticos
drift_monitor_log.csv	Historial de drift global