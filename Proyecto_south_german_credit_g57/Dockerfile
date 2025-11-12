# -----------------------------------------------------------------
# Fase 1: Base de Python
# -----------------------------------------------------------------
# Usamos una imagen de Python 3.10-slim como base. Es ligera y moderna.
FROM python:3.10-slim

# Establecemos el directorio de trabajo dentro del contenedor
WORKDIR /code

# -----------------------------------------------------------------
# Fase 2: Instalación de Dependencias
# -----------------------------------------------------------------
# Copiamos primero el archivo de requisitos
COPY ./requirements.txt /code/requirements.txt

# Instalamos las dependencias. Usamos --no-cache-dir para mantener
# la imagen ligera y --upgrade para obtener las últimas versiones.
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# -----------------------------------------------------------------
# Fase 3: Copia de la Aplicación
# -----------------------------------------------------------------
# Copiamos el script de inicio
COPY ./start.sh /code/start.sh

# Copiamos TODO el contenido de la carpeta 'app'
# (main.py, streamlit_app.py, y tu imagen a57.png)
COPY ./app /code/app

# -----------------------------------------------------------------
# Fase 4: Configuración Final y Ejecución
# -----------------------------------------------------------------
# Damos permisos de ejecución a nuestro script de inicio
RUN chmod +x /code/start.sh

# Exponemos ambos puertos para que sean accesibles desde fuera
EXPOSE 8000
EXPOSE 8001

# El comando final que se ejecuta cuando el contenedor arranca
CMD ["./start.sh"]