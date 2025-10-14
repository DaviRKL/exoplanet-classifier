"""Utilities for training and evaluating the exoplanet classifier.

Esta versão amplia o conjunto de features, melhora o pré‑processamento,
adiciona ensembles e interpretabilidade (SHAP) mantendo compatibilidade
com a API existente.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Any, Optional

import base64
import logging
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    balanced_accuracy_score,
    roc_auc_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Logger simples para treino
logger = logging.getLogger("exo.training")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

FEATURE_COLUMNS: List[str] = [
    # Flags de falso positivo
    "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec",
    # Orbitais/geométricas
    "koi_period", "koi_time0bk", "koi_impact", "koi_duration", "koi_depth",
    # Físicas/insolação
    "koi_prad", "koi_teq", "koi_insol", "koi_model_snr",
    # Estelares
    "koi_steff", "koi_slogg", "koi_srad",
    # Posicionais
    "ra", "dec", "koi_kepmag",
    # Novas variáveis do CSV
    "koi_score", "koi_pdisposition", "koi_tce_plnt_num",
    # Incertezas (err1) e derivadas SNR
    "koi_period_err1", "koi_duration_err1", "koi_depth_err1",
    "koi_prad_err1", "koi_steff_err1", "koi_srad_err1",
    # Derivadas criadas no pré-processamento
    "koi_duration_snr", "koi_depth_snr",
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

    # Campos extra opcionais (metadata que a API atual ignora se não existir)
    categorical_features: List[str] | None = None
    numeric_features: List[str] | None = None
    shap_summary: str | None = None  # imagem base64

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
                "categorical_features": self.categorical_features,
                "numeric_features": self.numeric_features,
                "shap_summary": self.shap_summary,
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

    label_encoder = LabelEncoder()

    df = pd.read_csv(path, comment="#")
    df_preprocessing = df[TARGET_COLUMN].astype(str).str.strip().str.upper()
    df[TARGET_COLUMN] = label_encoder.fit_transform(df_preprocessing)
    return df

def preprocess_dataset(df: pd.DataFrame, *, expect_target: bool = True) -> pd.DataFrame:
    """Normalize incoming data to the same format used by the training pipeline.

    - Renames known aliases (kepmag -> koi_kepmag)
    - Creates derived SNR columns when base + err1 are present
    - Coerces FEATURES to numeric; optional target normalization
    """
    df_processed = df.copy()

    if "kepmag" in df_processed.columns and "koi_kepmag" not in df_processed.columns:
        df_processed = df_processed.rename(columns={"kepmag": "koi_kepmag"})

    if all(col in df_processed.columns for col in ["koi_duration", "koi_duration_err1"]):
        with np.errstate(divide="ignore", invalid="ignore"):
            df_processed["koi_duration_snr"] = (
                pd.to_numeric(df_processed["koi_duration"], errors="coerce") /
                pd.to_numeric(df_processed["koi_duration_err1"], errors="coerce").abs()
            )
    if all(col in df_processed.columns for col in ["koi_depth", "koi_depth_err1"]):
        with np.errstate(divide="ignore", invalid="ignore"):
            df_processed["koi_depth_snr"] = (
                pd.to_numeric(df_processed["koi_depth"], errors="coerce") /
                pd.to_numeric(df_processed["koi_depth_err1"], errors="coerce").abs()
            )

    missing = [c for c in FEATURE_COLUMNS if c not in df_processed.columns]
    if missing:
        raise ValueError(f"Dataset is missing required feature columns: {missing}")

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

    df = df.copy()

    # Aliases comuns
    if "kepmag" in df.columns and "koi_kepmag" not in df.columns:
        df = df.rename(columns={"kepmag": "koi_kepmag"})

    # Cria variáveis derivadas (SNR) quando possível
    if all(col in df.columns for col in ["koi_duration", "koi_duration_err1"]):
        with np.errstate(divide="ignore", invalid="ignore"):
            df["koi_duration_snr"] = df["koi_duration"] / df["koi_duration_err1"].abs()
    if all(col in df.columns for col in ["koi_depth", "koi_depth_err1"]):
        with np.errstate(divide="ignore", invalid="ignore"):
            df["koi_depth_snr"] = df["koi_depth"] / df["koi_depth_err1"].abs()

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


def _split_feature_types() -> Tuple[List[str], List[str]]:
    """Divide features em numéricas e categóricas."""
    categorical_features = ["koi_pdisposition"]
    numeric_features = [c for c in FEATURE_COLUMNS if c not in categorical_features]
    return numeric_features, categorical_features


def _build_pipeline(estimator) -> Pipeline:
    """Compose preprocessing and the provided estimator into a pipeline."""

    numeric_features, categorical_features = _split_feature_types()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
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
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    matrix = confusion_matrix(y_test, y_pred, normalize=None)
    # Métricas adicionais para logs
    try:
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        logger.info("balanced_accuracy: %.3f", bal_acc)
    except Exception:
        pass
    try:
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(X_test)
        else:
            # Alguns pipelines expõem o método no estimador interno
            proba = pipeline.named_steps["classifier"].predict_proba(
                pipeline.named_steps["preprocessor"].transform(X_test)
            )
        roc = roc_auc_score(y_test, proba, multi_class="ovr")
        logger.info("roc_auc_ovr: %.3f", roc)
    except Exception:
        pass
    return TrainingMetrics(accuracy=accuracy, precision=precision, recall=recall, f1=f1, confusion=matrix, classes=classes)


def _fit_and_score(name: str, estimator: Any, X_train: pd.DataFrame, y_train: np.ndarray,
                   X_test: pd.DataFrame, y_test: np.ndarray, classes: List[str]) -> Tuple[str, Pipeline, TrainingMetrics, Dict[str, float]]:
    pipeline = _build_pipeline(estimator)
    logger.info("Training %s...", name)
    pipeline.fit(X_train, y_train)
    metrics = evaluate_model(pipeline, X_test, y_test, classes=classes)
    est = pipeline.named_steps["classifier"]
    importances = _compute_feature_importances(est, FEATURE_COLUMNS)
    return name, pipeline, metrics, importances


def _maybe_shap_summary(pipeline: Pipeline, X_sample: pd.DataFrame) -> Optional[str]:
    """Gera um summary plot SHAP (base64) se shap estiver disponível."""
    try:
        import shap  # type: ignore

        pre = pipeline.named_steps["preprocessor"]
        Xt = pre.transform(X_sample)
        try:
            feat_names = pre.get_feature_names_out()
        except Exception:
            feat_names = [f"f{i}" for i in range(Xt.shape[1])]

        explainer = shap.Explainer(pipeline.named_steps["classifier"])
        shap_values = explainer(Xt)
        plt.figure(figsize=(8, 5))
        shap.summary_plot(shap_values, feature_names=feat_names, show=False)
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")
    except Exception as exc:
        logger.info("SHAP summary unavailable: %s", exc)
        return None


def train_models(df: pd.DataFrame) -> Tuple[ModelBundle, Dict[str, TrainingMetrics]]:
    """Train multiple models and return the best bundle plus metrics map."""

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

    # Definição de modelos com hiperparâmetros mais fortes
    models: Dict[str, Any] = {
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_split=5,
            class_weight="balanced",
            criterion="entropy",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=600,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.2,
            eval_metric="mlogloss",
            random_state=42,
            tree_method="hist",
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
        ),
        "LogReg": LogisticRegression(max_iter=2000, n_jobs=-1, multi_class="auto", class_weight="balanced"),
        "SVC": SVC(C=2.0, gamma="scale", probability=True, class_weight="balanced", random_state=42),
    }

    metrics_map: Dict[str, TrainingMetrics] = {}
    model_results: Dict[str, TrainedModel] = {}

    trained: List[Tuple[str, Pipeline, TrainingMetrics, Dict[str, float]]] = []
    for name, est in models.items():
        res = _fit_and_score(name, est, X_train, y_train, X_test, y_test, list(label_encoder.classes_))
        trained.append(res)
        _, pipe, met, imps = res
        model_results[name] = TrainedModel(name=name, pipeline=pipe, metrics=met, feature_importances=imps)
        metrics_map[name] = met

    # Ensemble soft voting combinando os três melhores por F1
    top3 = sorted(trained, key=lambda t: t[2].f1, reverse=True)[:3]
    estimators = [(n, p.named_steps["classifier"]) for n, p, _, _ in top3]
    voting = VotingClassifier(estimators=estimators, voting="soft")
    name, pipe, met, imps = _fit_and_score("EnsembleVoting", voting, X_train, y_train, X_test, y_test, list(label_encoder.classes_))
    trained.append((name, pipe, met, imps))
    model_results[name] = TrainedModel(name=name, pipeline=pipe, metrics=met, feature_importances=imps)
    metrics_map[name] = met

    best_name = max(model_results.values(), key=lambda result: result.metrics.f1).name
    best_model = model_results[best_name]

    logger.info("Ranking (F1):")
    for n, _, m, _ in sorted(trained, key=lambda t: t[2].f1, reverse=True):
        logger.info("%s -> acc=%.3f f1=%.3f", n, m.accuracy, m.f1)

    # SHAP summary do melhor pipeline (amostra de treino)
    shap_b64 = _maybe_shap_summary(best_model.pipeline, X_train.head(500))

    num_feats, cat_feats = _split_feature_types()
    bundle = ModelBundle(
        label_encoder=label_encoder,
        feature_names=FEATURE_COLUMNS,
        model=best_model,
        categorical_features=cat_feats,
        numeric_features=num_feats,
        shap_summary=shap_b64,
    )
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

