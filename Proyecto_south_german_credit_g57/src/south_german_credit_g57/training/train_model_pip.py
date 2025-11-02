# ==========================================================
# TRAIN MODEL PIPELINE - SOUTH GERMAN CREDIT
# ==========================================================

# --- Imports base desde tu archivo de librerías ---
from south_german_credit_g57.libraries import *     

# --- Utilidades internas del proyecto ---
from south_german_credit_g57.utils.logger import get_logger
from south_german_credit_g57.evaluation.metrics import calculate_classification_metrics

# --- Configuración inicial ---
logger = get_logger("TrainModel")
warnings.filterwarnings("ignore")

# ==========================================================
# CLASE AUXILIAR
# ==========================================================
class BinaryEncoderWrapper(BaseEstimator, TransformerMixin):
    """Wrapper seguro para BinaryEncoder (convierte todo a string antes de codificar)."""
    def __init__(self, cols=None):
        self.cols = cols
        self.encoder = BinaryEncoder(cols=self.cols, return_df=True)

    def fit(self, X, y=None):
        X_ = X.copy().astype(str)
        self.encoder.fit(X_, y)
        return self

    def transform(self, X):
        X_ = X.copy().astype(str)
        return self.encoder.transform(X_)

# ==========================================================
# FUNCIONES BASE
# ==========================================================
def load_config(path: str) -> Dict:
    logger.info(f"Cargando configuración desde {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_data(path: str, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    logger.info(f"Leyendo dataset desde {path}")
    df = pd.read_csv(path)
    return df.drop(columns=[target_col]), df[target_col]

def create_preprocessor(config: Dict) -> ColumnTransformer:
    cfg = config["preprocessing"]
    logger.info("Creando pipeline de preprocesamiento...")

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=cfg["numeric"]["imputer_strategy"])),
        ("scaler", MinMaxScaler()),
        ("power", PowerTransformer(method="yeo-johnson"))
    ])
    nom_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=cfg["nominal"]["imputer_strategy"])),
        ("encoder", BinaryEncoderWrapper())
    ])
    ord_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=cfg["ordinal"]["imputer_strategy"])),
        ("scaler", MinMaxScaler())
    ])

    return ColumnTransformer([
        ("num", num_pipe, cfg["numeric"]["features"]),
        ("nom", nom_pipe, cfg["nominal"]["features"]),
        ("ord", ord_pipe, cfg["ordinal"]["features"])
    ], remainder="drop")

def get_model_class(name: str):
    models = {
        "LogisticRegression": LogisticRegression,
        "KNeighborsClassifier": KNeighborsClassifier,
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "RandomForest": RandomForestClassifier,
        "XGBoost": XGBClassifier,
        "MLP": MLPClassifier,
        "SVC": SVC
    }
    return models[name]

def get_sampler_class(name: str):
    samplers = {"SMOTE": SMOTE, "SMOTEENN": SMOTEENN, "NearMiss": NearMiss, "SMOTETomek": SMOTETomek}
    return samplers.get(name, None)

def get_scoring():
    gmean_scorer = make_scorer(geometric_mean_score, greater_is_better=True, average='binary')
    return {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score),
        "recall": make_scorer(recall_score),
        "f1": make_scorer(f1_score),
        "roc_auc": "roc_auc",
        "gmean": gmean_scorer
    }

# ==========================================================
# FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# ==========================================================
def main(config_path: str):
    config = load_config(config_path)
    X, y = load_data(config["data"]["train"], config["base"]["target_col"])
    preprocessor = create_preprocessor(config)

    # Configurar CV
    gs_cfg = config["grid_search"]
    cv = RepeatedStratifiedKFold(
        n_splits=gs_cfg["cv"],
        n_repeats=gs_cfg["n_repeats"],
        random_state=config["base"]["random_state"]
    )

    # Conexión MLflow
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    logger.info("Conectado a MLflow.")

    for key, model_cfg in config["training"].items():
        model_name = model_cfg["name"]
        logger.info(f"Entrenando modelo: {model_name}")

        with mlflow.start_run(run_name=model_name):
            model = get_model_class(model_name)()
            sampler_cls = get_sampler_class(model_cfg["resampler"])

            steps = [("preprocessor", preprocessor)]
            if sampler_cls:
                steps.append(("sampler", sampler_cls()))
            steps.append(("model", model))
            pipeline = ImbPipeline(steps=steps)

            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=model_cfg["param_grid"],
                scoring=gs_cfg["scoring"],
                cv=cv,
                n_jobs=gs_cfg["n_jobs"],
                verbose=gs_cfg["verbose"]
            )
            grid.fit(X, y)

            best_model = grid.best_estimator_
            best_score = grid.best_score_
            mlflow.log_params(grid.best_params_)
            mlflow.log_metric("best_cv_score", best_score)
            logger.info(f"Mejor score: {best_score:.4f}")

            # Re-evaluar
            scores = cross_validate(best_model, X, y, cv=cv, scoring=get_scoring())
            metrics_avg = {f"cv_{k.replace('test_','avg_')}": np.mean(v) for k, v in scores.items() if 'test_' in k}
            mlflow.log_metrics(metrics_avg)

            # Métricas finales
            y_pred = best_model.predict(X)
            y_prob = best_model.predict_proba(X)[:, 1] if hasattr(best_model, "predict_proba") else None
            custom_metrics = calculate_classification_metrics(y, y_pred, y_prob)
            mlflow.log_metrics({f"train_{k}": v for k, v in custom_metrics.items() if k != "confusion_matrix"})

            # Registrar modelo
            signature = infer_signature(X, best_model.predict(X))
            mlflow.sklearn.log_model(best_model, "model", signature=signature)
            logger.info(f"Modelo {model_name} registrado en MLflow.")

    logger.info("Entrenamiento completado para todos los modelos.")

# ==========================================================
# EJECUCIÓN DIRECTA
# ==========================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Entrenamiento y optimización de modelos.")
    parser.add_argument("--config", type=str, default="params.yaml", help="Ruta al archivo YAML.")
    args = parser.parse_args()
    main(args.config)
