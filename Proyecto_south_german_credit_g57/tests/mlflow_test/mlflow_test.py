import mlflow
import RandomPredictor as TestPredictor
from mlflow.models import infer_signature
from google.cloud import storage

EXPERIMENT_ID = "Experimento-Conexión-MLFlow-Grupo57"
MLFLOW_SERVER_URI_WITH_POSTGRE_SQL = "https://mlflow-g57-superior-137680020436.us-central1.run.app"

"""
    Prueba la conexión con el servidor MLFlow Remoto.
    No retorna ningún valor.
        
    Parameters
    ----------
    mlflow_uri: string
    La localización exacta del servidor MLFlow Remoto en la nube.
    
    experiment_id : string
        el ID único del experimento en MLFlow    

    Returns
    -------
    void
        No retorna valores.
"""


def test_mlflow_remote(mlflow_uri, experiment_id):
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_id)

    with mlflow.start_run() as run:
        predictor = TestPredictor.RandomPredictor()
        mlflow.log_params({})
        mlflow.log_metric("accuracy", 1.0)
        input_test = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
        signature = infer_signature(input_test, predictor.predict(context=None, model_input=input_test))

        model_info = mlflow.pyfunc.log_model(
            name="RandomPredictor",
            python_model=predictor,
            signature=signature,
            input_example=input_test,
            registered_model_name="tracking-grupo57",
        )

        mlflow.set_logged_model_tags(
            model_info.model_id,
            {"Training Info": "Prueba Exitosa"}
        )

    mlflow.end_run()


if __name__ == "__main__":
    storage_client = storage.Client(project="laboratorio1-447417")
    test_mlflow_remote(MLFLOW_SERVER_URI_WITH_POSTGRE_SQL, EXPERIMENT_ID)