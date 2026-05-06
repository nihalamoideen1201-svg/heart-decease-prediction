import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from train_model import train

EXPECTED_FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


def compute_file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_metadata(metadata_path: Path) -> dict[str, Any]:
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def validate_dataset(data_path: Path, target_column: str, min_rows: int) -> dict[str, Any]:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path.resolve()}")

    df = pd.read_csv(data_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset columns: {list(df.columns)}")

    feature_columns = [column for column in df.columns if column != target_column]
    missing = [column for column in EXPECTED_FEATURE_COLUMNS if column not in feature_columns]
    extra = [column for column in feature_columns if column not in EXPECTED_FEATURE_COLUMNS]

    if missing or extra:
        raise ValueError(
            "Dataset schema mismatch. "
            f"Missing feature columns: {missing or 'none'}. Extra feature columns: {extra or 'none'}."
        )

    if len(df) < min_rows:
        raise ValueError(f"Dataset has {len(df)} rows, which is below the minimum required {min_rows}.")

    return {
        "row_count": int(len(df)),
        "feature_columns": EXPECTED_FEATURE_COLUMNS,
        "dataset_hash": compute_file_hash(data_path),
    }


def should_retrain(dataset_hash: str, metadata: dict[str, Any], force: bool) -> bool:
    if force:
        return True
    return metadata.get("dataset_hash") != dataset_hash


def metrics_meet_thresholds(
    metrics: dict[str, Any],
    existing_metadata: dict[str, Any],
    min_accuracy: float,
    min_roc_auc: float,
    allow_regression: bool,
) -> tuple[bool, str]:
    accuracy = float(metrics["accuracy"])
    roc_auc = float(metrics.get("roc_auc", 0.0))

    if accuracy < min_accuracy:
        return False, f"Rejected candidate model: accuracy {accuracy:.4f} is below threshold {min_accuracy:.4f}."

    if "roc_auc" in metrics and roc_auc < min_roc_auc:
        return False, f"Rejected candidate model: roc_auc {roc_auc:.4f} is below threshold {min_roc_auc:.4f}."

    existing_accuracy = existing_metadata.get("metrics", {}).get("accuracy")
    existing_roc_auc = existing_metadata.get("metrics", {}).get("roc_auc")

    if not allow_regression and existing_accuracy is not None and accuracy < float(existing_accuracy):
        return (
            False,
            "Rejected candidate model: accuracy "
            f"{accuracy:.4f} is worse than current model accuracy {float(existing_accuracy):.4f}.",
        )

    if not allow_regression and existing_roc_auc is not None and "roc_auc" in metrics and roc_auc < float(existing_roc_auc):
        return (
            False,
            "Rejected candidate model: roc_auc "
            f"{roc_auc:.4f} is worse than current model roc_auc {float(existing_roc_auc):.4f}.",
        )

    return True, "Candidate model accepted."


def write_metadata(
    metadata_path: Path,
    *,
    data_path: Path,
    dataset_hash: str,
    target_column: str,
    row_count: int,
    metrics: dict[str, Any],
    model_path: Path,
) -> None:
    payload = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path.resolve()),
        "dataset_hash": dataset_hash,
        "target_column": target_column,
        "row_count": row_count,
        "feature_columns": EXPECTED_FEATURE_COLUMNS,
        "model_path": str(model_path.resolve()),
        "metrics": metrics,
    }
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def retrain_if_needed(
    *,
    data_path: Path,
    model_path: Path,
    metadata_path: Path,
    target_column: str,
    random_state: int,
    experiment_name: str,
    run_name: str | None,
    min_rows: int,
    min_accuracy: float,
    min_roc_auc: float,
    allow_regression: bool,
    force: bool,
) -> dict[str, Any]:
    existing_metadata = load_metadata(metadata_path)
    dataset_info = validate_dataset(data_path, target_column, min_rows)

    if not should_retrain(dataset_info["dataset_hash"], existing_metadata, force):
        return {
            "status": "skipped",
            "reason": "Dataset hash matches the current model metadata.",
            "dataset_hash": dataset_info["dataset_hash"],
        }

    candidate_model_path = model_path.with_name(f"{model_path.stem}_candidate{model_path.suffix}")
    try:
        metrics = train(
            data_path=data_path,
            model_path=candidate_model_path,
            target_column=target_column,
            random_state=random_state,
            experiment_name=experiment_name,
            run_name=run_name,
        )

        accepted, reason = metrics_meet_thresholds(
            metrics=metrics,
            existing_metadata=existing_metadata,
            min_accuracy=min_accuracy,
            min_roc_auc=min_roc_auc,
            allow_regression=allow_regression,
        )
        if not accepted:
            if candidate_model_path.exists():
                candidate_model_path.unlink()
            return {"status": "rejected", "reason": reason, "metrics": metrics}

        candidate_model_path.replace(model_path)
        write_metadata(
            metadata_path,
            data_path=data_path,
            dataset_hash=dataset_info["dataset_hash"],
            target_column=target_column,
            row_count=dataset_info["row_count"],
            metrics=metrics,
            model_path=model_path,
        )
        return {"status": "retrained", "reason": reason, "metrics": metrics, "dataset_hash": dataset_info["dataset_hash"]}
    finally:
        if candidate_model_path.exists():
            candidate_model_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain the model only when a new dataset is introduced.")
    parser.add_argument("--data", default="data/heart.csv", help="Path to the dataset CSV.")
    parser.add_argument("--model", default="model.pkl", help="Path to the production model artifact.")
    parser.add_argument("--metadata", default="model_metadata.json", help="Path to the model metadata JSON.")
    parser.add_argument("--target", default="target", help="Target column name in the dataset.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for the train/test split.")
    parser.add_argument(
        "--experiment-name",
        default="heart-disease-prediction",
        help="MLflow experiment name used to group retraining runs.",
    )
    parser.add_argument("--run-name", default=None, help="Optional MLflow run name.")
    parser.add_argument("--min-rows", type=int, default=50, help="Minimum row count required for retraining.")
    parser.add_argument("--min-accuracy", type=float, default=0.75, help="Minimum acceptable test accuracy.")
    parser.add_argument("--min-roc-auc", type=float, default=0.75, help="Minimum acceptable ROC AUC.")
    parser.add_argument(
        "--allow-regression",
        action="store_true",
        help="Allow the new model to replace the current one even if its metrics are worse.",
    )
    parser.add_argument("--force", action="store_true", help="Retrain even if the dataset hash has not changed.")
    args = parser.parse_args()

    result = retrain_if_needed(
        data_path=Path(args.data),
        model_path=Path(args.model),
        metadata_path=Path(args.metadata),
        target_column=args.target,
        random_state=args.random_state,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        min_rows=args.min_rows,
        min_accuracy=args.min_accuracy,
        min_roc_auc=args.min_roc_auc,
        allow_regression=args.allow_regression,
        force=args.force,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
