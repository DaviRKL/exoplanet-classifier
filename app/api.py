# app/api.py
"""FastAPI application exposing the exoplanet classifier and training utilities."""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.model_training import (
    FEATURE_COLUMNS,
    MODEL_PATH,
    ModelBundle,
    ModelResultSummary,
    generate_confusion_plot,
    generate_feature_importance_plot,
    load_dataset,
    load_model_bundle,
    preprocess_dataset,
    train_models,
    build_training_table,
)

# -----------------------------------------------------------------------------
# Logging & App configuration
# -----------------------------------------------------------------------------
logger = logging.getLogger("exoplanet_api")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

app = FastAPI(
    title="Exoplanet Classifier API",
    description="NASA Space Apps 2025 – Chubby Rockets",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Global state
# -----------------------------------------------------------------------------
CURRENT_BUNDLE: Optional[ModelBundle] = None


def _set_bundle(bundle: ModelBundle) -> None:
    global CURRENT_BUNDLE
    CURRENT_BUNDLE = bundle
    logger.info(
        "Loaded bundle for model %s (trained_at=%s)",
        bundle.best_model_name,
        bundle.trained_at,
    )


def _ensure_bundle_loaded() -> ModelBundle:
    if CURRENT_BUNDLE is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model before requesting predictions.",
        )
    return CURRENT_BUNDLE


def _load_existing_bundle() -> None:
    try:
        bundle = load_model_bundle(MODEL_PATH)
        _set_bundle(bundle)
    except FileNotFoundError:
        logger.warning("No persisted model found at %s yet.", MODEL_PATH)
    except KeyError as exc:
        logger.warning(
            "Persisted model is incompatible with the current schema (%s). Please retrain.",
            exc,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("Unable to load persisted model: %s", exc)


_load_existing_bundle()

# -----------------------------------------------------------------------------
# Pydantic schemas
# -----------------------------------------------------------------------------
class ExoplanetFeatures(BaseModel):
    koi_fpflag_nt: float = Field(..., description="False-positive flag: non-transit (0 or 1)")
    koi_fpflag_ss: float = Field(..., description="False-positive flag: stellar (0 or 1)")
    koi_fpflag_co: float = Field(..., description="False-positive flag: centroid offset (0 or 1)")
    koi_fpflag_ec: float = Field(..., description="False-positive flag: ephemeris match (0 or 1)")
    koi_period: float = Field(..., description="Orbital period (days)")
    koi_time0bk: float = Field(..., description="Transit epoch (BKJD)")
    koi_impact: float = Field(..., description="Impact parameter")
    koi_duration: float = Field(..., description="Transit duration (hours)")
    koi_depth: float = Field(..., description="Transit depth (fraction)")
    koi_prad: float = Field(..., description="Planet radius (Earth radii)")
    koi_teq: float = Field(..., description="Equilibrium temperature (K)")
    koi_insol: float = Field(..., description="Insolation flux (Earth flux)")
    koi_model_snr: float = Field(..., description="Model signal-to-noise ratio")
    koi_steff: float = Field(..., description="Stellar effective temperature (K)")
    koi_slogg: float = Field(..., description="Stellar surface gravity (log10(cm/s^2))")
    koi_srad: float = Field(..., description="Stellar radius (Solar radii)")
    ra: float = Field(..., description="Right ascension (deg)")
    dec: float = Field(..., description="Declination (deg)")
    kepmag: float = Field(..., description="Kepler magnitude")


class PredictionResponse(BaseModel):
    prediction: str


class ModelMetricsResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: List[List[int]]
    classes: List[str]
    classification_report: Optional[Dict[str, Any]] = None
    feature_importances: Optional[Dict[str, float]] = None


class TrainingResponse(BaseModel):
    best_model: str
    trained_at: Optional[str]
    training_seconds: Optional[float]
    dataset_info: Optional[Dict[str, int]]
    metrics: Dict[str, ModelMetricsResponse]


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float


class FeatureImportanceResponse(BaseModel):
    model: str
    top_features: List[FeatureImportanceItem]


class PlotResponse(BaseModel):
    confusion_matrix: str
    feature_importance: Optional[str]


# -----------------------------------------------------------------------------
# Helper conversions
# -----------------------------------------------------------------------------
def _summary_to_metrics(summary: ModelResultSummary) -> ModelMetricsResponse:
    metrics = summary.metrics
    return ModelMetricsResponse(
        accuracy=metrics.accuracy,
        precision=metrics.precision,
        recall=metrics.recall,
        f1=metrics.f1,
        confusion_matrix=metrics.confusion.tolist(),
        classes=metrics.classes,
        classification_report=summary.classification_report,
        feature_importances=summary.feature_importances,
    )


def _bundle_to_training_response(bundle: ModelBundle) -> TrainingResponse:
    metrics_payload = {
        name: _summary_to_metrics(summary) for name, summary in bundle.summaries.items()
    }
    return TrainingResponse(
        best_model=bundle.best_model_name,
        trained_at=bundle.trained_at,
        training_seconds=bundle.training_seconds,
        dataset_info=bundle.dataset_info,
        metrics=metrics_payload,
    )


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
def health_check() -> Dict[str, str]:
    """Basic health-check endpoint used by monitoring and clients."""
    loaded = CURRENT_BUNDLE is not None
    return {
        "message": "Exoplanet Classifier API is running 🚀",
        "model_loaded": str(loaded),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: ExoplanetFeatures) -> PredictionResponse:
    """Predict the KOI disposition given the submitted measurements."""
    bundle = _ensure_bundle_loaded()

    input_df = pd.DataFrame([features.dict()])
    try:
        processed = preprocess_dataset(input_df, expect_target=False)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    model_input = processed[bundle.feature_names]
    encoded_prediction = bundle.pipeline.predict(model_input)[0]
    class_label = bundle.label_encoder.inverse_transform([encoded_prediction])[0].upper()
    return PredictionResponse(prediction=class_label)


@app.post("/train", response_model=TrainingResponse)
async def train(file: UploadFile = File(...)) -> TrainingResponse:
    """Retrain the models using a new KOI catalog provided by the client."""
    try:
        contents = await file.read()
        raw_df = pd.read_csv(BytesIO(contents), comment="#")
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {exc}") from exc

    logger.info("Received dataset with %d rows for retraining.", len(raw_df))

    try:
        training_table = build_training_table(raw_df)
        bundle = train_models(training_table)
        bundle.save(MODEL_PATH)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected failure while training: %s", exc)
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}") from exc

    _set_bundle(bundle)
    return _bundle_to_training_response(bundle)


@app.get("/metrics", response_model=TrainingResponse)
def metrics() -> TrainingResponse:
    """Return evaluation metrics collected during the last training run."""
    bundle = _ensure_bundle_loaded()
    return _bundle_to_training_response(bundle)


@app.get("/feature-importance", response_model=FeatureImportanceResponse)
def feature_importance(top_n: int = Query(10, ge=1)) -> FeatureImportanceResponse:
    """Expose ordered feature importances for the current best model."""
    bundle = _ensure_bundle_loaded()
    summary = bundle.best_summary()

    if not summary.feature_importances:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{bundle.best_model_name}' does not expose feature importances.",
        )

    sorted_items = sorted(
        summary.feature_importances.items(), key=lambda item: item[1], reverse=True
    )[:top_n]
    items = [
        FeatureImportanceItem(feature=name, importance=float(value))
        for name, value in sorted_items
    ]
    return FeatureImportanceResponse(model=bundle.best_model_name, top_features=items)


@app.get("/plots", response_model=PlotResponse)
def plots(top_n: int = Query(10, ge=1)) -> PlotResponse:
    """Return base64 encoded plots for confusion matrix and feature importances."""
    bundle = _ensure_bundle_loaded()
    summary = bundle.best_summary()

    confusion_image = generate_confusion_plot(summary.metrics)

    fi_image: Optional[str] = None
    if summary.feature_importances:
        fi_image = generate_feature_importance_plot(summary.feature_importances, top_n=top_n)

    return PlotResponse(
        confusion_matrix=f"data:image/png;base64,{confusion_image}",
        feature_importance=f"data:image/png;base64,{fi_image}" if fi_image else None,
    )


# -----------------------------------------------------------------------------
# Utility endpoint (optional): re-load persisted bundle without retraining
# -----------------------------------------------------------------------------
@app.post("/reload")
def reload_bundle() -> TrainingResponse:
    """Reload model artifacts from disk (useful after manual training offline)."""
    try:
        bundle = load_model_bundle(MODEL_PATH)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No persisted model found on disk.")
    _set_bundle(bundle)
    return _bundle_to_training_response(bundle)
