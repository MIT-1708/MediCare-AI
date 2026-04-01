import json
import os
import subprocess
import warnings
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from keras_model import SerializableKerasBinaryClassifier

RANDOM_STATE = 42
DATA_DIR = "data"
EDA_DIR = "eda_outputs"
MODEL_DIR = "models"


@dataclass
class DatasetConfig:
    name: str
    csv_path: str
    target_col: str
    positive_label: str = ""


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_datasets() -> Dict[str, Tuple[pd.DataFrame, DatasetConfig]]:
    configs = {
        "heart": DatasetConfig(
            name="heart",
            csv_path=os.path.join(DATA_DIR, "heart.csv"),
            target_col="target",
        ),
        "diabetes": DatasetConfig(
            name="diabetes",
            csv_path=os.path.join(DATA_DIR, "diabetes.csv"),
            target_col="Outcome",
        ),
        "kidney": DatasetConfig(
            name="kidney",
            csv_path=os.path.join(DATA_DIR, "kidney_disease.csv"),
            target_col="classification",
            positive_label="ckd",
        ),
    }

    loaded: Dict[str, Tuple[pd.DataFrame, DatasetConfig]] = {}
    for key, cfg in configs.items():
        df = pd.read_csv(cfg.csv_path)
        loaded[key] = (df, cfg)
    return loaded


def preprocess_dataset(df: pd.DataFrame, cfg: DatasetConfig) -> Tuple[pd.DataFrame, pd.Series]:
    work = df.copy()

    if cfg.name == "kidney":
        if "id" in work.columns:
            work = work.drop(columns=["id"])
        for col in work.columns:
            if work[col].dtype == "object":
                work[col] = (
                    work[col]
                    .astype(str)
                    .str.strip()
                    .replace({"?": np.nan, "nan": np.nan})
                )
        work[cfg.target_col] = (
            work[cfg.target_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"ckd": 1, "notckd": 0})
        )
        work = work.dropna(subset=[cfg.target_col])

    X = work.drop(columns=[cfg.target_col])
    y = work[cfg.target_col]
    if y.dtype == "object":
        y = y.astype(str).str.strip().map({"1": 1, "0": 0}).fillna(y)
    y = pd.to_numeric(y, errors="coerce")

    valid_idx = y.notna()
    X = X.loc[valid_idx]
    y = y.loc[valid_idx].astype(int)

    # Keep numeric-like strings numeric where possible (especially for CKD)
    for col in X.columns:
        if X[col].dtype == "object":
            converted = pd.to_numeric(X[col], errors="coerce")
            numeric_ratio = converted.notna().mean()
            if numeric_ratio > 0.7:
                X[col] = converted
            else:
                X[col] = X[col].astype(str).str.strip().replace({"?": np.nan, "nan": np.nan})
    return X, y


def run_eda(X: pd.DataFrame, y: pd.Series, name: str) -> Dict[str, object]:
    _safe_mkdir(EDA_DIR)
    out = {
        "shape": X.shape,
        "dtypes": {k: str(v) for k, v in X.dtypes.to_dict().items()},
        "missing_values": {k: int(v) for k, v in X.isna().sum().to_dict().items()},
        "target_distribution": {
            str(k): float(v) for k, v in y.value_counts(normalize=True).to_dict().items()
        },
    }

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        corr = X[numeric_cols].corr(numeric_only=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm", center=0)
        plt.title(f"{name.upper()} Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_DIR, f"{name}_correlation_heatmap.png"))
        plt.close()

        X[numeric_cols].hist(figsize=(14, 10), bins=20)
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_DIR, f"{name}_feature_distributions.png"))
        plt.close()

    plt.figure(figsize=(5, 4))
    y.value_counts().plot(kind="bar", title=f"{name.upper()} Target Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, f"{name}_target_distribution.png"))
    plt.close()

    with open(os.path.join(EDA_DIR, f"{name}_eda_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )
    return preprocessor, num_cols, cat_cols


def maybe_apply_smote(X_train_np: np.ndarray, y_train: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    counts = y_train.value_counts()
    if len(counts) < 2:
        return X_train_np, y_train.values
    imbalance_ratio = counts.min() / counts.max()
    if imbalance_ratio < 0.65:
        sm = SMOTE(random_state=RANDOM_STATE)
        return sm.fit_resample(X_train_np, y_train.values)
    return X_train_np, y_train.values


def has_nvidia_gpu() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.returncode == 0 and "GPU" in (result.stdout or "")
    except Exception:
        return False


def get_models(use_gpu: bool = False):
    lr = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_split=2, random_state=RANDOM_STATE
    )
    
    gb_name = "gb"
    try:
        from xgboost import XGBClassifier

        xgb_params = dict(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        )
        if False:  # Force CPU to prevent Streamlit threading CUDA segfaults
            # XGBoost 2.x style GPU parameters.
            xgb_params["tree_method"] = "hist"
            xgb_params["device"] = "cuda"
        gb_model = XGBClassifier(**xgb_params)
        gb_name = "xgb"
    except Exception:
        warnings.warn("xgboost not available. Using GradientBoostingClassifier fallback.")
        gb_model = GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=7, random_state=RANDOM_STATE
        )
        
    from sklearn.ensemble import VotingClassifier
    voting = VotingClassifier(
        estimators=[('lr', lr), ('rf', rf), (gb_name, gb_model)],
        voting='soft'
    )
        
    models = {
        "logreg": lr,
        "rf": rf,
        gb_name: gb_model,
        "ensemble_voting": voting
    }
    keras_error = None
    try:
        import tensorflow  # noqa: F401

        models["keras_nn"] = SerializableKerasBinaryClassifier(
            input_dim=0,  # set after preprocessing size is known
            hidden_units=[64, 32],
            dropout_rate=0.15,
            learning_rate=1e-3,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            patience=6,
            random_state=RANDOM_STATE,
            verbose=0,
        )
    except Exception as exc:
        keras_error = str(exc)
    return models, keras_error


def evaluate_model(name: str, model, X_train, y_train, X_test, y_test) -> Dict[str, object]:
    cv_auc = None
    if name != "keras_nn":
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_auc = cross_val_score(model, X_train, y_train, scoring="roc_auc", cv=cv).mean()

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, proba)
    cm = confusion_matrix(y_test, pred)
    return {
        "model_name": name,
        "model": model,
        "cv_roc_auc": float(cv_auc) if cv_auc is not None else None,
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_precision": float(precision_score(y_test, pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, proba)),
        "confusion_matrix": cm.tolist(),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
    }


def select_best(results: List[Dict[str, object]]) -> Dict[str, object]:
    # Healthcare-priority ranking: recall first, then ROC-AUC
    ranked = sorted(results, key=lambda r: (r["test_recall"], r["test_roc_auc"]), reverse=True)
    return ranked[0]


def save_roc_plot(results: List[Dict[str, object]], disease: str) -> None:
    plt.figure(figsize=(7, 5))
    for r in results:
        fpr = r["roc_curve"]["fpr"]
        tpr = r["roc_curve"]["tpr"]
        aucv = r["test_roc_auc"]
        plt.plot(fpr, tpr, label=f"{r['model_name']} (AUC={aucv:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{disease.upper()} ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, f"{disease}_roc_curve.png"))
    plt.close()


def train_one(disease: str, X: pd.DataFrame, y: pd.Series, use_gpu: bool = False) -> Dict[str, object]:
    preprocessor, num_cols, _cat_cols = build_preprocessor(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    X_train_np = preprocessor.fit_transform(X_train)
    X_test_np = preprocessor.transform(X_test)
    feature_names_transformed = preprocessor.get_feature_names_out().tolist()

    X_train_bal, y_train_bal = maybe_apply_smote(X_train_np, y_train)
    models, keras_error = get_models(use_gpu=use_gpu)
    if "keras_nn" in models:
        models["keras_nn"].input_dim = int(X_train_bal.shape[1])
    if disease == "heart":
        if "keras_nn" not in models:
            raise RuntimeError(
                "Heart training is configured to use the Keras neural network only, "
                "but TensorFlow/Keras is not available in this environment."
            )
        models = {"keras_nn": models["keras_nn"]}

    results = []
    for model_name, model in models.items():
        eval_result = evaluate_model(
            model_name, model, X_train_bal, y_train_bal, X_test_np, y_test
        )
        results.append(eval_result)

    best = select_best(results)
    save_roc_plot(results, disease)
    summary = {
        "all_results": [
            {k: v for k, v in r.items() if k not in ["model", "roc_curve"]} for r in results
        ],
        "selected_model": best["model_name"],
        "selected_metrics": {k: v for k, v in best.items() if k not in ["model", "roc_curve"]},
        "means": {
            k: float(v) for k, v in X.select_dtypes(include=[np.number]).mean().to_dict().items()
        },
        "feature_columns": list(X.columns),
        "numeric_columns": num_cols,
        "positive_rate": float(y.mean()),
    }
    if keras_error:
        summary["keras_status"] = f"unavailable: {keras_error}"
    elif "keras_nn" in models:
        summary["keras_status"] = "available"
    return {
        "preprocessor": preprocessor,
        "best_model": best["model"],
        "summary": summary,
        "X_train_np": X_train_np,
        "X_test_np": X_test_np,
        "feature_names_transformed": feature_names_transformed,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train disease-specific models for risk prediction.")
    parser.add_argument(
        "--gpu",
        choices=["auto", "always", "never"],
        default="auto",
        help="GPU strategy for training. auto=use GPU if available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpu_available = has_nvidia_gpu()
    if args.gpu == "always":
        use_gpu = True
    elif args.gpu == "never":
        use_gpu = False
    else:
        use_gpu = gpu_available

    if use_gpu:
        print("GPU training enabled for XGBoost (if xgboost is installed).")
    else:
        print("GPU not enabled. Training on CPU.")

    _safe_mkdir(MODEL_DIR)
    _safe_mkdir(EDA_DIR)

    loaded = load_datasets()
    scalers = {}
    dataset_metadata = {}

    name_to_outfile = {
        "heart": "heart_model.pkl",
        "diabetes": "diabetes_model.pkl",
        "kidney": "kidney_model.pkl",
    }

    for disease, (df, cfg) in loaded.items():
        X, y = preprocess_dataset(df, cfg)
        _ = run_eda(X, y, disease)
        trained = train_one(disease, X, y, use_gpu=use_gpu)

        model_bundle = {
            "model": trained["best_model"],
            "preprocessor": trained["preprocessor"],
            "summary": trained["summary"],
            "feature_names_transformed": trained["feature_names_transformed"],
            "shap_background": trained["X_train_np"][:200],
        }

        joblib.dump(model_bundle, os.path.join(MODEL_DIR, name_to_outfile[disease]))
        scalers[disease] = trained["preprocessor"]
        dataset_metadata[disease] = trained["summary"]

        with open(os.path.join(MODEL_DIR, f"{disease}_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(trained["summary"], f, indent=2)

    joblib.dump(scalers, os.path.join(MODEL_DIR, "scalers.pkl"))
    with open(os.path.join(MODEL_DIR, "dataset_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_metadata, f, indent=2)

    print("Training complete. Saved:")
    print("- models/heart_model.pkl")
    print("- models/diabetes_model.pkl")
    print("- models/kidney_model.pkl")
    print("- models/scalers.pkl")


if __name__ == "__main__":
    main()
