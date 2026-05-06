import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train(
    data_path: Path,
    model_path: Path,
    target_column: str,
    random_state: int,
    experiment_name: str,
    run_name: str | None,
) -> dict:
    df = pd.read_csv(data_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset columns: {list(df.columns)}")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    params = {
        "model": "logistic_regression",
        "scaler": "standard_scaler",
        "solver": "liblinear",
        "max_iter": 1000,
        "random_state": random_state,
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("data_path", str(data_path.resolve()))
        mlflow.log_param("target_column", target_column)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("feature_count", X.shape[1])
        mlflow.log_params(params)

        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        solver=params["solver"],
                        max_iter=params["max_iter"],
                        random_state=params["random_state"],
                    ),
                ),
            ]
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
        }

        mlflow.log_metric("accuracy", metrics["accuracy"])

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
            mlflow.log_metric("roc_auc", metrics["roc_auc"])

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        metrics["cross_validation_accuracy"] = {
            "mean": float(cv_scores.mean()),
            "std": float(cv_scores.std()),
            "scores": [float(score) for score in cv_scores],
        }
        mlflow.log_metric("cv_accuracy_mean", metrics["cross_validation_accuracy"]["mean"])
        mlflow.log_metric("cv_accuracy_std", metrics["cross_validation_accuracy"]["std"])

        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path))

        report_path = model_path.with_name("training_report.json")
        report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(report_path))

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the heart disease model and log the run to MLflow.")
    parser.add_argument("--data", required=True, help="Path to the CSV dataset used for training.")
    parser.add_argument("--model", default="model.pkl", help="Path to write the trained model artifact.")
    parser.add_argument("--target", default="target", help="Target column name in the dataset.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for the train/test split.")
    parser.add_argument(
        "--experiment-name",
        default="heart-disease-prediction",
        help="MLflow experiment name used to group training runs.",
    )
    parser.add_argument("--run-name", default=None, help="Optional MLflow run name.")
    args = parser.parse_args()

    metrics = train(
        data_path=Path(args.data),
        model_path=Path(args.model),
        target_column=args.target,
        random_state=args.random_state,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
    )
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved model to {Path(args.model).resolve()}")


if __name__ == "__main__":
    main()
