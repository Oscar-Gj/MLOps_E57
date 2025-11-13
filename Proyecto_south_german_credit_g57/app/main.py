from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
from typing import Literal
import os

# Inicialización de FastAPI
app = FastAPI(title="Credit Prediction API")

# Ruta local del modelo
# MODEL_PATH = "models/latest_model.pkl" 
MODEL_PATH = "models/latest_model.pkl" 

# Variable global para el modelo
model = None

@app.on_event("startup")
async def load_model():
    """Cargar el modelo al iniciar la aplicación"""
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"No se encontró el archivo del modelo en: {MODEL_PATH}")
        
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        
        print(f"Modelo cargado exitosamente desde {MODEL_PATH}")
        print(f"Tipo de modelo: {type(model)}")
        
        # Verificar que el modelo tiene los métodos necesarios
        if not hasattr(model, 'predict'):
            raise AttributeError("El modelo no tiene el método 'predict'")
            
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        raise RuntimeError(f"Error al cargar el modelo desde {MODEL_PATH}: {e}")

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
    Nombres de columna y tipos de datos para predicción de crédito.
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

@app.get("/", tags=["Health Check"])
def read_root():
    """Endpoint de verificación de salud"""
    return {
        "status": "ok",
        "message": "Credit Prediction API está funcionando",
        "model_loaded": model is not None
    }

@app.get("/health", tags=["Health Check"])
def health_check():
    """Verificar el estado del modelo"""
    if model is None:
        raise HTTPException(status_code=503, detail="El modelo no está cargado")
    return {
        "status": "healthy",
        "model_type": str(type(model)),
        "model_path": MODEL_PATH
    }

@app.post("/predict", response_model=PredictionOutput, 
        summary="Realizar Predicción de Riesgo",
        tags=["Predicciones"])
def predict(data: CreditInput):
    """
    Realizar predicción de riesgo crediticio
    
    - **Entrada**: Datos del solicitante de crédito
    - **Salida**: Etiqueta de riesgo (Alto/Bajo), valor de predicción y probabilidad
    """
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Error del Servidor: El modelo no está cargado."
        )

    try:
        # Convertir a DataFrame con las columnas en el orden esperado
        input_dict = data.dict()
        input_df = pd.DataFrame([{col: float(input_dict[col]) for col in EXPECTED_COLUMNS}])

        # Predicción
        prediction = model.predict(input_df)
        prediction_value = int(prediction[0])

        # Probabilidad (solo si el modelo la tiene)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)
            # La probabilidad de la clase predicha
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

    except KeyError as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Falta una columna requerida: {str(e)}"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Error en el formato de los datos: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error interno durante la predicción: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)