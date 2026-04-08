import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split


def evaluate(data_path: Path, model_path: Path, target_column: str, random_state: int) -> dict:
    df = pd.read_csv(data_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset columns: {list(df.columns)}")

    X = df.drop(columns=[target_column])
    y = df[target_column]
    model = joblib.load(model_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_prob)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    metrics["cross_validation_accuracy"] = {
        "mean": float(cv_scores.mean()),
        "std": float(cv_scores.std()),
        "scores": [float(score) for score in cv_scores],
    }

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the saved heart disease model.")
    parser.add_argument("--data", required=True, help="Path to a CSV dataset used for evaluation.")
    parser.add_argument("--model", default="model.pkl", help="Path to the saved model artifact.")
    parser.add_argument("--target", default="target", help="Target column name in the dataset.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for train/test split.")
    parser.add_argument("--output", default="evaluation_report.json", help="Path to the JSON report output.")
    args = parser.parse_args()

    report = evaluate(Path(args.data), Path(args.model), args.target, args.random_state)
    output_path = Path(args.output)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved evaluation report to {output_path.resolve()}")


if __name__ == "__main__":
    main()
