"""FastAPI application exposing the exoplanet classifier and training utilities."""
from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
import logging
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.model_training import (
    FEATURE_COLUMNS,
    MODEL_PATH,
    TrainingMetrics,
    build_training_table,
    generate_confusion_plot,
    generate_feature_importance_plot,
    preprocess_dataset,
    train_models,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("exoplanet.api")

# Inform FastAPI that the app is served behind a reverse proxy under the 
# path prefix "/api". This ensures the Swagger UI points to /api/openapi.json
# while the upstream keeps serving /openapi.json (nginx strips the prefix).
app = FastAPI(title="Exoplanet Classifier API", version="2.0.0", root_path="/api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        'http://localhost:4200',
        'http://127.0.0.1:4200',
        'http://localhost:8080',
        'http://127.0.0.1:8080',
    ],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


MODEL_PIPELINE = None
LABEL_ENCODER = None
FEATURE_NAMES: List[str] = FEATURE_COLUMNS.copy()
BEST_MODEL_NAME: str | None = None
TRAINING_RESULTS: Dict[str, TrainingMetrics] = {}
FEATURE_IMPORTANCES: Dict[str, float] = {}


class ExoplanetFeatures(BaseModel):
    """Input schema aligned with training features.

    All fields are optional to allow partial payloads; missing values are
    imputed by the pipeline's SimpleImputer during preprocessing.
    """

    koi_fpflag_nt: Optional[float] = Field(None, description="Non-transit FP flag (0/1)")
    koi_fpflag_ss: Optional[float] = Field(None, description="Stellar FP flag (0/1)")
    koi_fpflag_co: Optional[float] = Field(None, description="Centroid offset FP flag (0/1)")
    koi_fpflag_ec: Optional[float] = Field(None, description="Ephemeris match FP flag (0/1)")
    koi_period: Optional[float] = Field(None, description="Orbital period (days)")
    koi_time0bk: Optional[float] = Field(None, description="Transit epoch (BKJD)")
    koi_impact: Optional[float] = Field(None, description="Impact parameter")
    koi_duration: Optional[float] = Field(None, description="Transit duration (hours)")
    koi_depth: Optional[float] = Field(None, description="Transit depth (fraction)")
    koi_prad: Optional[float] = Field(None, description="Planet radius (Earth radii)")
    koi_teq: Optional[float] = Field(None, description="Equilibrium temperature (K)")
    koi_insol: Optional[float] = Field(None, description="Insolation flux (Earth=1)")
    koi_model_snr: Optional[float] = Field(None, description="Model SNR")
    koi_steff: Optional[float] = Field(None, description="Stellar Teff (K)")
    koi_slogg: Optional[float] = Field(None, description="log g (cm/s^2)")
    koi_srad: Optional[float] = Field(None, description="Stellar radius (Rsun)")
    ra: Optional[float] = Field(None, description="Right ascension (deg)")
    dec: Optional[float] = Field(None, description="Declination (deg)")
    koi_kepmag: Optional[float] = Field(None, description="Kepler magnitude")


class PredictionResponse(BaseModel):
    """Prediction returned by the classifier."""

    prediction: str


class ModelMetricsResponse(BaseModel):
    """Serialized metrics for a trained classifier."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: List[List[int]]
    classes: List[str]


class TrainingResponse(BaseModel):
    """Payload returned after retraining the model."""

    best_model: str
    metrics: Dict[str, ModelMetricsResponse]


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float


class FeatureImportanceResponse(BaseModel):
    model: str
    importances: List[FeatureImportanceItem]


class PlotResponse(BaseModel):
    """Base64 encoded visualisations for dashboards."""

    confusion_matrix: str
    feature_importance: str


def _load_artifacts() -> None:
    """Load persisted artifacts if they exist on disk."""

    global MODEL_PIPELINE, LABEL_ENCODER, FEATURE_NAMES, BEST_MODEL_NAME, TRAINING_RESULTS, FEATURE_IMPORTANCES

    if not MODEL_PATH.exists():
        logger.info("MODEL_PATH %s not found. API will start without a trained model.", MODEL_PATH)
        return

    bundle = joblib.load(MODEL_PATH)
    MODEL_PIPELINE = bundle.get("pipeline")
    LABEL_ENCODER = bundle.get("label_encoder")
    FEATURE_NAMES = bundle.get("feature_names", FEATURE_COLUMNS)
    BEST_MODEL_NAME = bundle.get("model_name")
    metrics = bundle.get("metrics")
    if isinstance(metrics, TrainingMetrics):
        TRAINING_RESULTS = {BEST_MODEL_NAME: metrics}
    else:
        TRAINING_RESULTS = {}
    FEATURE_IMPORTANCES = bundle.get("feature_importances", {})
    logger.info("Loaded model artifacts: model=%s features=%d", BEST_MODEL_NAME, len(FEATURE_NAMES))


def _ensure_model_loaded() -> None:
    if MODEL_PIPELINE is None or LABEL_ENCODER is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model before predicting.")


def _serialize_metrics(metrics: TrainingMetrics) -> ModelMetricsResponse:
    confusion_list = metrics.confusion.astype(int).tolist()
    return ModelMetricsResponse(
        accuracy=metrics.accuracy,
        precision=metrics.precision,
        recall=metrics.recall,
        f1=metrics.f1,
        confusion_matrix=confusion_list,
        classes=metrics.classes,
    )


@app.get("/")
def health_check() -> dict[str, Any]:
    """Basic health-check endpoint used by monitoring and clients."""

    return {
        "message": "Exoplanet Classifier API is running",
        "model_loaded": MODEL_PIPELINE is not None,
    }

@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"status": "ok", "model_loaded": MODEL_PIPELINE is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(features: ExoplanetFeatures) -> PredictionResponse:
    """Predict the KOI disposition given the submitted measurements."""

    _ensure_model_loaded()

    def _to_float(v: Any) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            v = v.strip().replace(",", ".")
            try:
                return float(v)
            except ValueError:
                return None
        return None

    raw = features.model_dump() if hasattr(features, "model_dump") else features.dict()
    row = {name: _to_float(raw.get(name)) for name in FEATURE_NAMES}
    model_input = pd.DataFrame([row])

    try:
        processed = preprocess_dataset(model_input, expect_target=False)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        encoded_prediction = MODEL_PIPELINE.predict(processed[FEATURE_NAMES])
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}") from exc

    class_label = LABEL_ENCODER.inverse_transform(encoded_prediction)[0].upper()
    return PredictionResponse(prediction=class_label)


@app.post("/train", response_model=TrainingResponse)
async def train(file: UploadFile = File(...)) -> TrainingResponse:
    """Retrain the models using a new KOI catalog provided by the client."""

    try:
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents), comment="#")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {exc}")

    try:
        training_table = build_training_table(df)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    bundle, metrics_map = train_models(training_table)
    bundle.save(MODEL_PATH)

    _apply_bundle(bundle, metrics_map)

    serialized = {name: _serialize_metrics(metric) for name, metric in metrics_map.items()}
    return TrainingResponse(best_model=BEST_MODEL_NAME, metrics=serialized)


@app.get("/metrics", response_model=TrainingResponse)
def metrics() -> TrainingResponse:
    """Return evaluation metrics collected during the last training run."""

    if not TRAINING_RESULTS or BEST_MODEL_NAME is None:
        raise HTTPException(status_code=404, detail="No training metrics available yet.")

    serialized = {name: _serialize_metrics(metric) for name, metric in TRAINING_RESULTS.items()}
    return TrainingResponse(best_model=BEST_MODEL_NAME, metrics=serialized)


@app.get("/feature-importance", response_model=FeatureImportanceResponse)
def feature_importance(top_n: int = Query(10, ge=1)) -> FeatureImportanceResponse:
    """Expose ordered feature importances for the current model."""

    _ensure_model_loaded()

    sorted_items = sorted(FEATURE_IMPORTANCES.items(), key=lambda item: item[1], reverse=True)[:top_n]
    response_items = [FeatureImportanceItem(feature=feature, importance=value) for feature, value in sorted_items]
    return FeatureImportanceResponse(model=BEST_MODEL_NAME or "unknown", importances=response_items)


@app.get("/plots", response_model=PlotResponse)
def plots(top_n: int = Query(10, ge=1)) -> PlotResponse:
    """Return base64 encoded plots for confusion matrix and feature importances."""

    if not TRAINING_RESULTS or BEST_MODEL_NAME is None:
        raise HTTPException(status_code=404, detail="Train the model before requesting plots.")

    metrics = TRAINING_RESULTS[BEST_MODEL_NAME]
    confusion_image = generate_confusion_plot(metrics)
    importance_image = generate_feature_importance_plot(FEATURE_IMPORTANCES, top_n=top_n)
    return PlotResponse(
        confusion_matrix=f"data:image/png;base64,{confusion_image}",
        feature_importance=f"data:image/png;base64,{importance_image}",
    )


def _apply_bundle(bundle, metrics_map: Dict[str, TrainingMetrics]) -> None:
    """Update in-memory state once new artifacts are computed."""

    global MODEL_PIPELINE, LABEL_ENCODER, FEATURE_NAMES, BEST_MODEL_NAME, TRAINING_RESULTS, FEATURE_IMPORTANCES

    MODEL_PIPELINE = bundle.model.pipeline
    LABEL_ENCODER = bundle.label_encoder
    FEATURE_NAMES = bundle.feature_names
    BEST_MODEL_NAME = bundle.model.name
    TRAINING_RESULTS = metrics_map
    FEATURE_IMPORTANCES = bundle.model.feature_importances


@app.on_event("startup")
def on_startup() -> None:
    logger.info("Starting Exoplanet Classifier API...")
    try:
        _load_artifacts()
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to load persisted artifacts: %s", exc)
    logger.info("Startup complete. model_loaded=%s", MODEL_PIPELINE is not None)

@app.on_event("shutdown")
def on_shutdown() -> None:
    logger.info("Shutting down Exoplanet Classifier API.")








