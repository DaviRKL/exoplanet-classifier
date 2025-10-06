# app/model_training.py
"""Utilities for training and evaluating the exoplanet classifier."""

from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
logger = logging.getLogger("exoplanet_training")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# Domain-specific settings
# -----------------------------------------------------------------------------
FEATURE_COLUMNS: List[str] = [
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
    "koi_period",
    "koi_time0bk",
    "koi_impact",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_teq",
    "koi_insol",
    "koi_model_snr",
    "koi_steff",
    "koi_slogg",
    "koi_srad",
    "ra",
    "dec",
    "koi_kepmag",
]
FLAG_COLUMNS = ["koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"]
ALIAS_COLUMNS: Dict[str, List[str]] = {
    "koi_kepmag": ["kepmag"],
}
TARGET_COLUMN = "koi_disposition"
EXPECTED_CLASSES = ["CANDIDATE", "CONFIRMED", "FALSE POSITIVE"]

MODEL_PATH = Path("models/exoplanet_classifier.joblib")
DEFAULT_DATA_PATH = Path("koi-cumulative.csv")

# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------
@dataclass
class TrainingMetrics:
    """Aggregated metrics for a trained classifier."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: np.ndarray
    classes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": float(self.accuracy),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1": float(self.f1),
            "confusion_matrix": self.confusion.tolist(),
            "classes": self.classes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingMetrics":
        return cls(
            accuracy=float(data["accuracy"]),
            precision=float(data["precision"]),
            recall=float(data["recall"]),
            f1=float(data["f1"]),
            confusion=np.array(data["confusion_matrix"]),
            classes=list(data["classes"]),
        )


@dataclass
class ModelResultSummary:
    """Store evaluation outputs for a single trained model."""

    metrics: TrainingMetrics
    classification_report: Dict[str, Any]
    feature_importances: Optional[Dict[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metrics": self.metrics.to_dict(),
            "classification_report": self.classification_report,
            "feature_importances": self.feature_importances,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelResultSummary":
        return cls(
            metrics=TrainingMetrics.from_dict(data["metrics"]),
            classification_report=data.get("classification_report", {}),
            feature_importances=data.get("feature_importances"),
        )


@dataclass
class ModelBundle:
    """Artifacts persisted so the API can serve predictions and dashboards."""

    label_encoder: LabelEncoder
    feature_names: List[str]
    best_model_name: str
    pipeline: Pipeline
    summaries: Dict[str, ModelResultSummary]
    trained_at: str
    training_seconds: float
    dataset_info: Dict[str, Any]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "label_encoder": self.label_encoder,
            "feature_names": self.feature_names,
            "best_model_name": self.best_model_name,
            "pipeline": self.pipeline,
            "summaries": {name: summary.to_dict() for name, summary in self.summaries.items()},
            "trained_at": self.trained_at,
            "training_seconds": self.training_seconds,
            "dataset_info": self.dataset_info,
        }
        joblib.dump(payload, path)

    def best_summary(self) -> ModelResultSummary:
        return self.summaries[self.best_model_name]


# -----------------------------------------------------------------------------
# Data preparation helpers
# -----------------------------------------------------------------------------
def estimate_stellar_mass(koi_srad: float | None, koi_slogg: float | None) -> float | None:
    """Estimate stellar mass (in solar masses) when it is missing in the catalog."""
    if koi_srad is None or koi_slogg is None or np.isnan(koi_srad) or np.isnan(koi_slogg):
        return None
    if koi_srad <= 0:
        return None

    log_g_sun = 4.438  # Surface gravity of the Sun in log10(cm/s^2).
    log_mass = koi_slogg - log_g_sun + 2 * np.log10(koi_srad)
    mass = float(10 ** log_mass)
    if not np.isfinite(mass):
        return None
    return mass


def load_dataset(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the KOI cumulative catalog and standardize the disposition labels."""
    df = pd.read_csv(path, comment="#")
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip().str.upper()
    return df


def preprocess_dataset(df: pd.DataFrame, *, expect_target: bool = True) -> pd.DataFrame:
    """Ensure all required feature columns exist and are numeric."""
    df_processed = df.copy()

    # Remap known aliases (e.g., kepmag -> koi_kepmag) before validation
    for canonical, aliases in ALIAS_COLUMNS.items():
        if canonical not in df_processed.columns:
            for alias in aliases:
                if alias in df_processed.columns:
                    df_processed = df_processed.rename(columns={alias: canonical})
                    break

    missing_features = [col for col in FEATURE_COLUMNS if col not in df_processed.columns]
    if missing_features:
        raise ValueError(f"Dataset is missing required feature columns: {missing_features}")

    for col in FEATURE_COLUMNS:
        df_processed[col] = pd.to_numeric(df_processed[col], errors="coerce")

    if expect_target:
        if TARGET_COLUMN not in df_processed.columns:
            raise ValueError(f"Dataset must contain the target column '{TARGET_COLUMN}'.")
        df_processed[TARGET_COLUMN] = (
            df_processed[TARGET_COLUMN].astype(str).str.strip().str.upper()
        )

    return df_processed
def build_training_table(df: pd.DataFrame) -> pd.DataFrame:
    """Project the raw catalog onto the columns required by the models."""
    df = preprocess_dataset(df, expect_target=True)

    if "koi_smass" not in df.columns:
        df["koi_smass"] = df.apply(
            lambda row: estimate_stellar_mass(row.get("koi_srad"), row.get("koi_slogg")),
            axis=1,
        )

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns after preprocessing: {missing}")

    table = df[required_columns].replace([np.inf, -np.inf], np.nan)
    table.dropna(subset=[TARGET_COLUMN], inplace=True)

    return table


def _build_pipeline(estimator: Any) -> Pipeline:
    """Compose preprocessing and the provided estimator into a scikit-learn pipeline."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, FEATURE_COLUMNS)],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )


def _compute_feature_importances(estimator: Any, feature_names: Iterable[str]) -> Optional[Dict[str, float]]:
    """Retrieve feature importances if the estimator exposes them."""
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return None
    return {name: float(score) for name, score in zip(feature_names, importances)}


def evaluate_model_detailed(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
) -> Tuple[TrainingMetrics, Dict[str, Dict[str, float]]]:
    """Calculate aggregate metrics, confusion matrix and per-class report."""
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    matrix = confusion_matrix(y_test, y_pred)

    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    report = classification_report(
        y_test_labels,
        y_pred_labels,
        labels=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )

    metrics = TrainingMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        confusion=matrix,
        classes=list(label_encoder.classes_),
    )
    return metrics, report


# -----------------------------------------------------------------------------
# Training routine
# -----------------------------------------------------------------------------
def train_models(df: pd.DataFrame) -> ModelBundle:
    """Train Random Forest and XGBoost models, returning the bundle of artifacts."""
    processed = preprocess_dataset(df, expect_target=True)

    unknown_classes = sorted(set(processed[TARGET_COLUMN].unique()) - set(EXPECTED_CLASSES))
    if unknown_classes:
        raise ValueError(f"Unexpected disposition classes found: {unknown_classes}")

    label_encoder = LabelEncoder()
    label_encoder.fit(EXPECTED_CLASSES)

    X = processed[FEATURE_COLUMNS]
    y = label_encoder.transform(processed[TARGET_COLUMN])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    dataset_info = {
        "n_rows": int(len(processed)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "n_features": len(FEATURE_COLUMNS),
    }

    estimators: Dict[str, Any] = {
        "RandomForest": RandomForestClassifier(
            n_estimators=600,
            max_depth=16,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            use_label_encoder=False,
            eval_metric="mlogloss",
            tree_method="hist",
            booster="gbtree",
            n_estimators=700,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.7,
            reg_lambda=1.0,
            reg_alpha=0.0,
            min_child_weight=3,
            gamma=0.0,
            random_state=42,
        ),
    }

    start_time = time.perf_counter()
    summaries: Dict[str, ModelResultSummary] = {}
    best_model_name = ""
    best_f1 = -np.inf
    best_pipeline: Optional[Pipeline] = None

    for name, estimator in estimators.items():
        logger.info("Training %s on %d samples...", name, len(X_train))
        pipeline = _build_pipeline(estimator)
        pipeline.fit(X_train, y_train)

        metrics, report = evaluate_model_detailed(pipeline, X_test, y_test, label_encoder)
        importances = _compute_feature_importances(pipeline.named_steps["classifier"], FEATURE_COLUMNS)

        summaries[name] = ModelResultSummary(
            metrics=metrics,
            classification_report=report,
            feature_importances=importances,
        )

        logger.info(
            "Results %s -> accuracy: %.3f | precision: %.3f | recall: %.3f | f1: %.3f",
            name,
            metrics.accuracy,
            metrics.precision,
            metrics.recall,
            metrics.f1,
        )

        weighted_report = report.get("weighted avg", {})
        logger.debug("Weighted report for %s: %s", name, weighted_report)

        if metrics.f1 > best_f1:
            best_f1 = metrics.f1
            best_model_name = name
            best_pipeline = pipeline

    if best_pipeline is None:
        raise RuntimeError("No model was successfully trained.")

    training_seconds = time.perf_counter() - start_time
    trained_at = datetime.utcnow().isoformat(timespec="seconds")

    logger.info(
        "Best model: %s (f1=%.3f) trained in %.2f seconds",
        best_model_name,
        best_f1,
        training_seconds,
    )

    bundle = ModelBundle(
        label_encoder=label_encoder,
        feature_names=FEATURE_COLUMNS,
        best_model_name=best_model_name,
        pipeline=best_pipeline,
        summaries=summaries,
        trained_at=trained_at,
        training_seconds=training_seconds,
        dataset_info=dataset_info,
    )
    return bundle


def train_and_save(
    data_path: Path = DEFAULT_DATA_PATH,
    model_path: Path = MODEL_PATH,
) -> ModelBundle:
    """High level helper that mirrors the legacy CLI behaviour."""
    dataset = load_dataset(data_path)
    training_table = build_training_table(dataset)
    bundle = train_models(training_table)
    bundle.save(model_path)
    return bundle


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------
def generate_confusion_plot(metrics: TrainingMetrics) -> str:
    """Render a confusion matrix heatmap and return it as a base64 string."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        metrics.confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=metrics.classes,
        yticklabels=metrics.classes,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    buffer = BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


def generate_feature_importance_plot(importances: Dict[str, float], top_n: int = 10) -> str:
    """Render a bar chart for feature importances and return it as a base64 string."""
    sorted_items = sorted(importances.items(), key=lambda item: item[1], reverse=True)[:top_n]
    labels = [item[0] for item in sorted_items]
    scores = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=scores, y=labels, palette="viridis", ax=ax)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title(f"Top {len(labels)} Feature Importances")
    buffer = BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


# -----------------------------------------------------------------------------
# Persistence helpers
# -----------------------------------------------------------------------------
def load_model_bundle(path: Path = MODEL_PATH) -> ModelBundle:
    """Load a previously trained bundle from disk with backwards compatibility."""

    payload = joblib.load(path)

    if "summaries" in payload:
        summaries = {
            name: ModelResultSummary.from_dict(summary)
            for name, summary in payload["summaries"].items()
        }
        best_model_name = payload["best_model_name"]
    else:
        legacy_name = payload.get("model_name", "RandomForest")
        legacy_metrics = payload.get("metrics")
        if isinstance(legacy_metrics, TrainingMetrics):
            metrics = legacy_metrics
        elif isinstance(legacy_metrics, dict):
            metrics = TrainingMetrics.from_dict(legacy_metrics)
        else:
            raise KeyError(
                "Legacy bundle missing 'metrics' information; please retrain the model."
            )

        feature_importances = payload.get("feature_importances")
        summaries = {
            legacy_name: ModelResultSummary(
                metrics=metrics,
                classification_report={},
                feature_importances=feature_importances,
            )
        }
        best_model_name = legacy_name

    bundle = ModelBundle(
        label_encoder=payload["label_encoder"],
        feature_names=list(payload["feature_names"]),
        best_model_name=best_model_name,
        pipeline=payload["pipeline"],
        summaries=summaries,
        trained_at=payload.get("trained_at", ""),
        training_seconds=float(payload.get("training_seconds", 0.0)),
        dataset_info=payload.get("dataset_info", {}),
    )
    return bundle

__all__ = [
    "FEATURE_COLUMNS",
    "FLAG_COLUMNS",
    "TARGET_COLUMN",
    "EXPECTED_CLASSES",
    "MODEL_PATH",
    "TrainingMetrics",
    "ModelResultSummary",
    "ModelBundle",
    "load_dataset",
    "preprocess_dataset",
    "build_training_table",
    "train_models",
    "train_and_save",
    "generate_confusion_plot",
    "generate_feature_importance_plot",
    "load_model_bundle",
]

