# Usa una versión de Python correcta para el despliegue
FROM python:3.10-slim-bullseye

# Instala MLFlow y otras dependencias recomendadas
RUN pip install --no-cache mlflow==3.1.1 psycopg2-binary boto3

# Establece el directorio de trabajo
WORKDIR /app

# Abre el puerto MLflow respectivo
EXPOSE 5000

# Inicia el servicio para servir MLFlow
COPY start.sh /app/start.sh

RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]