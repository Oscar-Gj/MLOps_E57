import numpy as np
import random
import os

def set_seed(seed: int = 42):
    """Fija la semilla global para reproducibilidad."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
