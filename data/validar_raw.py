import pandas as pd
import os
from datetime import datetime
import dvc.api

class RawDataValidator:
    """
    Clase para validar, renombrar y estandarizar datasets crudos (Raw Layer).
    Compatible con pipelines MLOps versionados con DVC.
    """

    def __init__(self, ruta_raw, ruta_diccionario, ruta_validado):
        self.ruta_raw = ruta_raw
        self.ruta_diccionario = ruta_diccionario
        self.ruta_validado = ruta_validado
        self.df = None
        self.mapping = None

    def leer_diccionario(self):
        """Carga el diccionario de columnas esperado."""
        if not os.path.exists(self.ruta_diccionario):
            raise FileNotFoundError(f"No se encontró el diccionario: {self.ruta_diccionario}")

        df_dict = pd.read_csv(self.ruta_diccionario)
        if not {"original", "nuevo"}.issubset(df_dict.columns):
            raise ValueError("El CSV de diccionario debe tener columnas: 'original' y 'nuevo'")

        self.mapping = dict(zip(df_dict["original"], df_dict["nuevo"]))
        print(f"Diccionario cargado correctamente ({len(self.mapping)} columnas mapeadas).")

    def leer_raw(self):
        """Carga el dataset crudo desde la capa Raw."""
        if not os.path.exists(self.ruta_raw):
            raise FileNotFoundError(f"No se encontró el archivo Raw: {self.ruta_raw}")

        self.df = pd.read_csv(self.ruta_raw)
        print(f"Archivo Raw leído: {os.path.basename(self.ruta_raw)}")
        print(f"Columnas detectadas: {list(self.df.columns)}")

    def validar_columnas(self):
        """Verifica columnas faltantes o sobrantes respecto al diccionario."""
        esperadas = set(self.mapping.keys())
        presentes = set(self.df.columns)
        faltantes = esperadas - presentes
        sobrantes = presentes - esperadas

        if faltantes:
            print(f"Faltan columnas: {faltantes}")
        if sobrantes:
            print(f"Sobran columnas: {sobrantes}")
        if not faltantes and not sobrantes:
            print("Todas las columnas esperadas están presentes.")

        return faltantes, sobrantes

    def registrar_log(self, faltantes, sobrantes):
        """Registra en log la validación y el hash DVC del dataset Raw."""
        os.makedirs(os.path.dirname(self.ruta_validado), exist_ok=True)
        log_path = os.path.join(os.path.dirname(self.ruta_validado), "log_validacion.txt")

        # Obtener hash o URL del dataset desde DVC
        try:
            raw_version = dvc.api.get_url(self.ruta_raw, repo=".")
        except Exception:
            raw_version = "No disponible (ejecutado localmente)"

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n--- Validación ejecutada: {datetime.now()} ---\n")
            f.write(f"Archivo RAW: {self.ruta_raw}\n")
            f.write(f"Versión DVC/Hash: {raw_version}\n")
            f.write(f"Faltantes: {faltantes if faltantes else 'Ninguna'}\n")
            f.write(f"Sobrantes: {sobrantes if sobrantes else 'Ninguna'}\n")
        print(f"Log actualizado en: {log_path}")

    def renombrar(self):
        """Estandariza nombres de columnas según el diccionario."""
        self.df.rename(columns=self.mapping, inplace=True)
        print("Columnas renombradas correctamente.")
        print(f"Nuevas columnas: {list(self.df.columns)}")

    def guardar_validado(self):
        """Guarda el archivo validado (para la capa Validated)."""
        os.makedirs(os.path.dirname(self.ruta_validado), exist_ok=True)
        self.df.to_csv(self.ruta_validado, index=False)
        print(f"Archivo validado guardado en: {self.ruta_validado}")

    def ejecutar(self):
        """Pipeline completo: Raw → Validated."""
        self.leer_diccionario()
        self.leer_raw()
        faltantes, sobrantes = self.validar_columnas()
        self.registrar_log(faltantes, sobrantes)
        self.renombrar()
        self.guardar_validado()

if __name__ == "__main__":
    project_root = os.getcwd()  # raíz del proyecto donde está dvc.yaml
    ruta_raw = os.path.join(project_root, "data", "raw", "german_credit_modified.csv")
    ruta_diccionario = os.path.join(project_root, "data", "diccionario_columnas.csv")
    ruta_validado = os.path.join(project_root, "data", "interim", "german_credit_validado.csv")

    validador = RawDataValidator(ruta_raw, ruta_diccionario, ruta_validado)
    validador.ejecutar()


