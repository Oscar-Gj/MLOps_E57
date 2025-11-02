# ============================================
# Módulo: config.py
# --------------------------------------------
# Objetivo:
# - Fijar la semilla global para reproducibilidad
# - Centralizar la configuración básica del proyecto
# ============================================

import numpy as np
import random
import os

# ==== Configuración general ====
RANDOM_STATE = 57  # Usa el mismo número en todos tus scripts

def set_seed(seed: int = RANDOM_STATE):
    """
    Fija la semilla global para garantizar reproducibilidad en todo el flujo:
    - Numpy
    - random
    - Python hash
    """
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Semilla global fijada en {seed}")

def get_random_state() -> int:
    """
    Devuelve el valor actual de RANDOM_STATE.
    Útil para pasar a modelos o validaciones cruzadas.
    """
    return RANDOM_STATE
