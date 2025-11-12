from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow
import mlflow.pyfunc
from typing import Literal

# Inicialización de FastAPI
app = FastAPI(title="Credit Prediction API")

# Ruta o URI del modelo en MLflow
# model_uri = "runs:/3998d4c5b6174664b586ce09c170bbbd/model"
model_name = "LogisticRegression_model"
model_alias = "best"
model_uri = f"models:/{model_name}@{model_alias}"

# Cargar el modelo desde MLflow
try:
    mlflow.set_tracking_uri("https://mlflow-super-g57-137680020436.us-central1.run.app")
    model = mlflow.pyfunc.load_model(model_uri)
    print("✅ Modelo cargado exitosamente desde MLflow")
except Exception as e:
    raise RuntimeError(f"Error al cargar el modelo desde MLflow: {e}")

# Columnas esperadas por el modelo
EXPECTED_COLUMNS = [
    "status", "duration", "credit_history", "purpose", "amount",
    "savings", "employment_duration", "installment_rate", "personal_status_sex",
    "other_debtors", "present_residence", "property", "age", "other_installment_plans",
    "housing", "number_credits", "job", "people_liable", "telephone", "foreign_worker"
]

# Estructura de entrada (para FastAPI)
class CreditInput(BaseModel):
    """
    Schema de ENTRADA. 
    ¡Nombres de columna y tipos de datos CORREGIDOS!
    """
    status: float
    duration: float
    credit_history: float
    purpose: float
    amount: float
    savings: float
    employment_duration: float
    installment_rate: float
    personal_status_sex: float
    other_debtors: float
    present_residence: float
    property: float
    age: float
    other_installment_plans: float
    housing: float
    number_credits: float
    job: float
    people_liable: float
    telephone: float
    foreign_worker: float

    class Config:
        schema_extra = {
            "example": {
                "status": 4.0,
                "duration": 12.0,
                "credit_history": 4.0,
                "purpose": 3.0,
                "amount": 1934.0,
                "savings": 1.0,
                "employment_duration": 5.0,
                "installment_rate": 2.0,
                "personal_status_sex": 3.0,
                "other_debtors": 1.0,
                "present_residence": 2.0,
                "property": 4.0,
                "age": 26.0,
                "other_installment_plans": 3.0,
                "housing": 2.0,
                "number_credits": 2.0,
                "job": 3.0,
                "people_liable": 2.0,
                "telephone": 1.0,
                "foreign_worker": 2.0
            }
        }
class PredictionOutput(BaseModel):
    """Schema de SALIDA (Traducido)."""
    etiqueta_prediccion: Literal["Riesgo Bajo", "Riesgo Alto"]
    valor_prediccion: int
    probabilidad: float

@app.post("/predict", response_model=PredictionOutput, 
        summary="Realizar Predicción de Riesgo",
        tags=["Predicciones"])
def predict(data: CreditInput):
    if model is None:
        raise HTTPException(status_code=503, 
                            detail="Error del Servidor: El modelo no está cargado.")

    try:
        # Convertir a DataFrame
        input_df = pd.DataFrame([{k: float(v) for k, v in data.dict().items()}])

        # Predicción
        prediction = model.predict(input_df)
        prediction_value = int(prediction[0])

        # Probabilidad (solo si el modelo la tiene)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)
            probability = float(probabilities[0][prediction_value])
        else:
            # Valor por defecto si no existe predict_proba
            probability = 1.0 if prediction_value == 1 else 0.0

        # Etiqueta legible
        label = "Riesgo Alto" if prediction_value == 1 else "Riesgo Bajo"

        return PredictionOutput(
            etiqueta_prediccion=label,
            valor_prediccion=prediction_value,
            probabilidad=probability
        )

    except Exception as e:
        raise HTTPException(status_code=500, 
                            detail=f"Error interno durante la predicción: {str(e)}")
