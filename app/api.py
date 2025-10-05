"""FastAPI application exposing the exoplanet classifier and training utilities."""
from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.model_training import (
    FEATURE_COLUMNS,
    MODEL_PATH,
    TrainingMetrics,
    build_training_table,
    generate_confusion_plot,
    generate_feature_importance_plot,
    train_models,
)

app = FastAPI(title="Exoplanet Classifier API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:4200', 'http://127.0.0.1:4200'],
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
    """Input schema that mirrors the columns used by the model."""

    koi_period: float = Field(..., description="Orbital period in days")
    koi_duration: float = Field(..., description="Transit duration in hours")
    koi_depth: float = Field(..., description="Transit depth (fraction of stellar flux)")
    koi_prad: float = Field(..., description="Planetary radius in Earth radii")
    koi_steff: float = Field(..., description="Stellar Effective Temperature")
    koi_srad: float = Field(..., description="Stellar radius in solar radii")


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

@app.post("/predict", response_model=PredictionResponse)
def predict(features: ExoplanetFeatures) -> PredictionResponse:
    """Predict the KOI disposition given the submitted measurements."""

    _ensure_model_loaded()

    ordered_values = [getattr(features, name) for name in FEATURE_NAMES]
    model_input = pd.DataFrame([ordered_values], columns=FEATURE_NAMES, dtype=float)

    try:
        encoded_prediction = MODEL_PIPELINE.predict(model_input)
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


try:
    _load_artifacts()
except Exception as exc:  # pragma: no cover - defensive
    print(f"Failed to load persisted artifacts: {exc}")
