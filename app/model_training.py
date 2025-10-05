"""Utilities for training and evaluating the exoplanet classifier."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import base64
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
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

FEATURE_COLUMNS: List[str] = [
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_steff",
    "koi_srad"
]
TARGET_COLUMN = "koi_disposition"

MODEL_PATH = Path("models/exoplanet_classifier.joblib")
DEFAULT_DATA_PATH = Path("koi-cumulative.csv")


@dataclass
class TrainingMetrics:
    """Aggregated metrics for a trained classifier."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: np.ndarray
    classes: List[str]

    def as_dict(self) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass
class TrainedModel:
    """Container for a single trained classifier."""

    name: str
    pipeline: Pipeline
    metrics: TrainingMetrics
    feature_importances: Dict[str, float]


@dataclass
class ModelBundle:
    """Artifacts persisted with joblib so the API can load them."""

    label_encoder: LabelEncoder
    feature_names: List[str]
    model: TrainedModel

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "label_encoder": self.label_encoder,
                "feature_names": self.feature_names,
                "model_name": self.model.name,
                "pipeline": self.model.pipeline,
                "metrics": self.model.metrics,
                "feature_importances": self.model.feature_importances,
            },
            path,
        )


# def estimate_stellar_mass(koi_srad: float | None, koi_slogg: float | None) -> float | None:
#     """Estimate stellar mass (in solar masses) when it is missing in the catalog."""

#     if koi_srad is None or koi_slogg is None or np.isnan(koi_srad) or np.isnan(koi_slogg):
#         return None
#     if koi_srad <= 0:
#         return None

#     log_g_sun = 4.438  # Surface gravity of the Sun in log10(cm/s^2).
#     log_mass = koi_slogg - log_g_sun + 2 * np.log10(koi_srad)
#     mass = float(10 ** log_mass)
#     if not np.isfinite(mass):
#         return None
#     return mass


def load_dataset(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the KOI cumulative catalog and standardize the disposition labels."""

    df = pd.read_csv(path, comment="#")
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip().str.upper()
    return df


def build_training_table(df: pd.DataFrame) -> pd.DataFrame:
    """Project the raw catalog onto the columns required by the models."""

    df = df.copy()

    # if "koi_smass" not in df.columns:
    #     df["koi_smass"] = df.apply(
    #         lambda row: estimate_stellar_mass(row.get("koi_srad"), row.get("koi_slogg")),
    #         axis=1,
    #     )

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    table = df[required_columns].replace([np.inf, -np.inf], np.nan)
    table.dropna(subset=[TARGET_COLUMN], inplace=True)

    return table


def _build_pipeline(estimator) -> Pipeline:
    """Compose preprocessing and the provided estimator into a pipeline."""

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
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


def _compute_feature_importances(estimator, feature_names: Iterable[str]) -> Dict[str, float]:
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return {name: 0.0 for name in feature_names}
    return {name: float(score) for name, score in zip(feature_names, importances)}


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray, classes: List[str]) -> TrainingMetrics:
    """Calculate aggregate metrics and confusion matrix for a fitted pipeline."""

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    matrix = confusion_matrix(y_test, y_pred)
    return TrainingMetrics(accuracy=accuracy, precision=precision, recall=recall, f1=f1, confusion=matrix, classes=classes)


def train_models(df: pd.DataFrame) -> Tuple[ModelBundle, Dict[str, TrainingMetrics]]:
    """Train Random Forest and XGBoost models, returning the best one."""

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(df[TARGET_COLUMN])
    X = df[FEATURE_COLUMNS]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    candidates = {
        "RandomForest": _build_pipeline(RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")),
        "XGBoost": _build_pipeline(
            XGBClassifier(
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=42,
                n_estimators=400,
                learning_rate=0.10,
                max_depth=6,
            )
        ),
    }

    metrics_map: Dict[str, TrainingMetrics] = {}
    model_results: Dict[str, TrainedModel] = {}

    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test, classes=list(label_encoder.classes_))
        estimator = pipeline.named_steps["classifier"]
        feature_importances = _compute_feature_importances(estimator, FEATURE_COLUMNS)
        model_results[name] = TrainedModel(name=name, pipeline=pipeline, metrics=metrics, feature_importances=feature_importances)
        metrics_map[name] = metrics

    best_name = max(model_results.values(), key=lambda result: result.metrics.f1).name
    best_model = model_results[best_name]

    bundle = ModelBundle(label_encoder=label_encoder, feature_names=FEATURE_COLUMNS, model=best_model)
    return bundle, metrics_map


def generate_confusion_plot(metrics: TrainingMetrics) -> str:
    """Render a confusion matrix heatmap and return it as a base64 string."""

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(metrics.confusion, annot=True, fmt="d", cmap="Blues", xticklabels=metrics.classes, yticklabels=metrics.classes, ax=ax)
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
    ax.set_title("Top Feature Importances")
    buffer = BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


def train_and_save(
    data_path: Path = DEFAULT_DATA_PATH,
    model_path: Path = MODEL_PATH,
) -> ModelBundle:
    """High level helper that mirrors the legacy CLI behaviour."""

    dataset = load_dataset(data_path)
    training_table = build_training_table(dataset)
    bundle, _ = train_models(training_table)
    bundle.save(model_path)
    return bundle


if __name__ == "__main__":
    saved_bundle = train_and_save()
    print(f"Model saved to {MODEL_PATH.resolve()}")
    print(f"Best model: {saved_bundle.model.name}")
