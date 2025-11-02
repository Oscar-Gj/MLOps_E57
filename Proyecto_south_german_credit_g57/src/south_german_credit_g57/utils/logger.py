# ==========================================================
# LOGGER CONFIGURATION - SOUTH GERMAN CREDIT PROJECT
# ==========================================================
from south_german_credit_g57.libraries import *   # usa tus librerías globales
from datetime import datetime

def get_logger(name: str = "MainLogger") -> logging.Logger:
    """
    Configura un logger con salida tanto en consola como en archivo.

    Parámetros:
        name (str): nombre del logger (ej. 'TrainModel', 'MainPipeline')
    Retorna:
        logging.Logger configurado.
    """
    # ------------------------------------------
    # 1️⃣ Definir ruta del directorio de logs
    # ------------------------------------------
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Archivo de log por fecha (YYYY-MM-DD.log)
    log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log")

    # ------------------------------------------
    # 2️⃣ Crear el logger
    # ------------------------------------------
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # evita duplicación de mensajes en consola

    # ------------------------------------------
    # 3️⃣ Evitar duplicar handlers
    # ------------------------------------------
    if not logger.handlers:
        # Formato del mensaje
        formatter = logging.Formatter(
            "%(asctime)s — [%(levelname)s] — %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Handler de consola
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Handler de archivo (rotativo por día)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Mensaje inicial
        logger.info(f" Logger '{name}' inicializado — Archivo: {os.path.basename(log_file)}")

    return logger
