import pandas as pd
import numpy as np
import yaml
import argparse
import logging
import mlflow
import mlflow.models
from mlflow.tracking import MlflowClient  
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score, f1_score, precision_score, recall_score
from imblearn.metrics import geometric_mean_score 

# --- Configuración del Logger ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


import chardet, yaml, pathlib

def load_config(config_path: str):
    path = pathlib.Path(config_path)
    raw = path.read_bytes()
    enc_info = chardet.detect(raw)
    detected = enc_info.get("encoding", "utf-8")
    if detected.lower() != "utf-8":
        text = raw.decode(detected, errors="ignore")
        path.write_text(text, encoding="utf-8")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_test_data(path: str, target_col: str):
    """Carga los datos de prueba y los divide en X, y."""
    logger.info(f"Cargando datos de prueba desde: {path}")
    try:
        df = pd.read_csv(path)
        X_test = df.drop(columns=[target_col])
        y_test = df[target_col]
        logger.info(f"Datos de prueba cargados. Shape de X: {X_test.shape}, Shape de y: {y_test.shape}")
        return X_test, y_test
    except FileNotFoundError:
        logger.error(f"Error: No se encontró el archivo de datos en {path}.")
        sys.exit(1)

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Genera y guarda la matriz de confusión."""
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Bueno (0)', 'Malo (1)'], 
                    yticklabels=['Bueno (0)', 'Malo (1)'])
        plt.title('Matriz de Confusión')
        plt.ylabel('Valor Real')
        plt.xlabel('Predicción')
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Matriz de confusión guardada en: {save_path}")
    except Exception as e:
        logger.warning(f"No se pudo generar la matriz de confusión: {e}")

def plot_roc_curve(y_true, y_proba, save_path):
    """Genera y guarda la curva ROC."""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title('Curva ROC')
        plt.legend(loc="lower right")
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Curva ROC guardada en: {save_path}")
    except Exception as e:
        logger.warning(f"No se pudo generar la curva ROC: {e}")

def find_best_model(config: dict) -> str:
    """
    Busca en MLflow el mejor run del experimento de entrenamiento 
    y devuelve el nombre del modelo registrado correspondiente.
    """
    mlflow_config = config.get('mlflow', {})
    train_experiment_name = mlflow_config.get('experiment_name', 'Experimento-Conexión-MLFlow-Grupo57')
    metric_name = config['grid_search']['scoring']
    sort_metric = f"metrics.avg_cv_{metric_name}"
    
    logger.info(f"Buscando el mejor modelo en el experimento '{train_experiment_name}' ordenado por '{sort_metric}' DESC")
    
    try:
        best_runs = mlflow.search_runs(
            experiment_names=[train_experiment_name],
            order_by=[f"{sort_metric} DESC"],
            max_results=1
        )
        
        if best_runs.empty:
            logger.error(f"No se encontraron runs en el experimento '{train_experiment_name}' que tengan la métrica '{sort_metric}'.")
            logger.error("Asegúrate de haber ejecutado 'train_model.py' (versión GridSearchCV) y que haya finalizado exitosamente.")
            sys.exit(1)
            
        best_run = best_runs.iloc[0]
        best_run_name = best_run["tags.mlflow.runName"]
        best_score = best_run[sort_metric]
        
        logger.info(f"Mejor run de entrenamiento encontrado: '{best_run_name}' con {sort_metric.split('.')[-1]} = {best_score:.4f}")
        
        if "_GridSearch" not in best_run_name:
            logger.error(f"El nombre del run '{best_run_name}' no sigue el formato esperado 'ModelName_GridSearch'.")
            sys.exit(1)
            
        base_model_name = best_run_name.replace("_GridSearch", "")
        registered_model_name = f"{base_model_name}_model"
        
        logger.info(f"Modelo 'campeón' identificado: '{registered_model_name}'")
        return registered_model_name

    except Exception as e:
        logger.error(f"Error al buscar el mejor modelo en MLflow: {e}", exc_info=True)
        sys.exit(1)

def main(config_path: str, model_name: str):
    
    # 1. Cargar Configuración y Datos
    config = load_config(config_path)
    X_test, y_test = load_test_data(
        path=config['data']['test'],
        target_col=config['base']['target_col']
    )
    
    # 2. Configurar MLflow
    mlflow_config = config.get('mlflow', {})
    if 'tracking_uri' in mlflow_config:
        mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
    eval_experiment_name = mlflow_config.get('evaluation_experiment_name', 'MLFlow-Grupo57-Evaluación')
    mlflow.set_experiment(eval_experiment_name)
    
    # 3. Determinar el nombre del modelo
    if model_name is None:
        model_name = find_best_model(config)
    
    logger.info(f"--- Iniciando Evaluación para el modelo: {model_name} ---")

    # 4. Crear directorios de reportes
    reports_dir = Path("reports/evaluation")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # 5. Iniciar Run de Evaluación en MLflow
    with mlflow.start_run(run_name=f"eval_{model_name}_v-latest") as run:
        run_id = run.info.run_id
        logger.info(f"Run de MLflow iniciado (ID: {run_id}). Evaluando modelo en {eval_experiment_name}")
        
        # 6. Definir URIs y Cargar Información del Modelo
        model_version = "latest" 
        model_uri = f"models:/{model_name}/{model_version}"
        
        # Registrar qué modelo se está evaluando
        mlflow.log_param("model_name_evaluated", model_name)
        mlflow.log_param("model_version_evaluated", model_version)
        mlflow.log_param("model_uri_evaluated", model_uri)

        try:
            logger.info(f"Obteniendo información del run de entrenamiento para {model_uri}")
            
            # Obtener el run_id del modelo registrado
            model_info = mlflow.models.get_model_info(model_uri)
            parent_run_id = model_info.run_id
            mlflow.log_param("parent_training_run_id", parent_run_id)

            # Conectar al cliente de MLflow
            client = MlflowClient()
            parent_run_data = client.get_run(parent_run_id).data

            # Registrar todos los parámetros del run de entrenamiento
            logger.info("Registrando parámetros del run de entrenamiento...")
            parent_params = parent_run_data.params
            for k, v in parent_params.items():
                # Prefijar para evitar colisiones y dar claridad
                mlflow.log_param(f"train_{k}", v) 

            # Registrar todas las métricas del run de entrenamiento (ej. avg_cv_metrics)
            logger.info("Registrando métricas del run de entrenamiento...")
            parent_metrics = parent_run_data.metrics
            for k, v in parent_metrics.items():
                # Prefijar para evitar colisiones y dar claridad
                mlflow.log_metric(f"train_{k}", v)
            
            logger.info("Información del entrenamiento registrada exitosamente.")

            # 7. Cargar el artefacto del modelo
            logger.info(f"Cargando el artefacto del modelo desde: {model_uri}")
            loaded_model = mlflow.sklearn.load_model(model_uri)
            
        except Exception as e:
            logger.error(f"Error al cargar el modelo o sus metadatos '{model_name}' desde MLflow: {e}", exc_info=True)
            mlflow.end_run(status="FAILED")
            sys.exit(1)
            
        # 8. Generar predicciones
        logger.info("Iniciando predicciones en el conjunto de prueba...")
        y_pred = loaded_model.predict(X_test)
        y_proba = loaded_model.predict_proba(X_test)[:, 1] # Probabilidad de la clase '1'
        
        # 9. Calcular y registrar métricas de TEST
        metrics = {
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred),
            "test_recall": recall_score(y_test, y_pred),
            "test_f1": f1_score(y_test, y_pred),
            "test_roc_auc": roc_auc_score(y_test, y_proba),
            "test_gmean": geometric_mean_score(y_test, y_pred)
        }
        
        mlflow.log_metrics(metrics)
        logger.info("Métricas de PRUEBA (test set) registradas:")
        for k, v in metrics.items():
            logger.info(f"{k}: {v:.4f}")
            
        # 10. Generar y registrar artefactos (gráficos)
        cm_path = str(reports_dir / f"cm_{run_id}.png")
        roc_path = str(reports_dir / f"roc_{run_id}.png")
        
        plot_confusion_matrix(y_test, y_pred, cm_path)
        plot_roc_curve(y_test, y_proba, roc_path)
        
        mlflow.log_artifact(cm_path, "plots")
        mlflow.log_artifact(roc_path, "plots")
        
        # 11. Registrar reporte de clasificación
        report_str = classification_report(y_test, y_pred, target_names=['Bueno (0)', 'Malo (1)'])
        report_path = str(reports_dir / f"report_{run_id}.txt")
        with open(report_path, 'w') as f:
            f.write(report_str)
        
        mlflow.log_artifact(report_path)
        logger.info(f"Reporte de clasificación guardado y registrado: {report_path}")
        logger.info("Evaluación finalizada y registrada en MLflow.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de evaluación de modelo sobre el conjunto de prueba.")
    parser.add_argument('--config', type=str, default='params.yaml', help='Ruta al archivo de configuración YAML.')
    
    parser.add_argument('--model_name', type=str, required=False, default=None, 
                        help='(Opcional) Nombre del modelo en MLflow. Si no se provee, busca el mejor del experimento de training.')
    
    args = parser.parse_args()
    main(config_path=args.config, model_name=args.model_name)