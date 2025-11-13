#!/bin/bash

# Manejo de limpieza al salir
cleanup() {
  echo "Deteniendo servicios..."
  pkill -f "uvicorn app.main:app" 2>/dev/null
  pkill -f "streamlit run app/streamlit_app.py" 2>/dev/null
  exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# 1. Lanza la API de FastAPI en segundo plano
echo "Iniciando FastAPI (Uvicorn) en el puerto 8000..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# 2. Lanza la App de Streamlit en primer plano
echo "Iniciando Streamlit en el puerto 8001..."
streamlit run app/streamlit_app.py --server.port 8001 --server.address 0.0.0.0