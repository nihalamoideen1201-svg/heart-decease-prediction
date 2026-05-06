import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import retrain_if_needed


class RetrainIfNeededTests(unittest.TestCase):
    def setUp(self) -> None:
        self.test_root = Path(__file__).resolve().parent / "_tmp_retrain"
        if self.test_root.exists():
            shutil.rmtree(self.test_root)
        self.test_root.mkdir(parents=True)

    def tearDown(self) -> None:
        if self.test_root.exists():
            shutil.rmtree(self.test_root)

    def test_validate_dataset_rejects_missing_feature_columns(self) -> None:
        data_path = self.test_root / "heart.csv"
        data_path.write_text(
            "age,sex,target\n63,1,1\n58,0,0\n61,1,1\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "Dataset schema mismatch"):
            retrain_if_needed.validate_dataset(data_path, "target", min_rows=2)

    def test_retrain_skips_when_dataset_hash_matches_metadata(self) -> None:
        data_path = self.test_root / "heart.csv"
        metadata_path = self.test_root / "model_metadata.json"
        model_path = self.test_root / "model.pkl"
        rows = [
            "age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target",
            "63,1,3,145,233,1,0,150,0,2.3,0,0,1,1",
            "37,1,2,130,250,0,1,187,0,3.5,0,0,2,1",
            "41,0,1,130,204,0,0,172,0,1.4,2,0,2,1",
            "57,0,0,140,192,0,1,148,0,0.4,1,0,1,0",
        ]
        data_path.write_text("\n".join(rows), encoding="utf-8")
        dataset_hash = retrain_if_needed.compute_file_hash(data_path)
        metadata_path.write_text(json.dumps({"dataset_hash": dataset_hash}), encoding="utf-8")

        result = retrain_if_needed.retrain_if_needed(
            data_path=data_path,
            model_path=model_path,
            metadata_path=metadata_path,
            target_column="target",
            random_state=42,
            experiment_name="test",
            run_name=None,
            min_rows=4,
            min_accuracy=0.7,
            min_roc_auc=0.7,
            allow_regression=False,
            force=False,
        )

        self.assertEqual(result["status"], "skipped")

    def test_retrain_rejects_candidate_below_thresholds(self) -> None:
        data_path = self.test_root / "heart.csv"
        metadata_path = self.test_root / "model_metadata.json"
        model_path = self.test_root / "model.pkl"
        rows = [
            "age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target",
            "63,1,3,145,233,1,0,150,0,2.3,0,0,1,1",
            "37,1,2,130,250,0,1,187,0,3.5,0,0,2,1",
            "41,0,1,130,204,0,0,172,0,1.4,2,0,2,1",
            "57,0,0,140,192,0,1,148,0,0.4,1,0,1,0",
        ]
        data_path.write_text("\n".join(rows), encoding="utf-8")

        def fake_train(**kwargs):
            Path(kwargs["model_path"]).write_text("candidate", encoding="utf-8")
            return {"accuracy": 0.6, "roc_auc": 0.65}

        with patch("retrain_if_needed.train", side_effect=fake_train):
            result = retrain_if_needed.retrain_if_needed(
                data_path=data_path,
                model_path=model_path,
                metadata_path=metadata_path,
                target_column="target",
                random_state=42,
                experiment_name="test",
                run_name=None,
                min_rows=4,
                min_accuracy=0.7,
                min_roc_auc=0.7,
                allow_regression=False,
                force=False,
            )

        self.assertEqual(result["status"], "rejected")
        self.assertFalse(model_path.exists())

    def test_retrain_promotes_candidate_and_writes_metadata(self) -> None:
        data_path = self.test_root / "heart.csv"
        metadata_path = self.test_root / "model_metadata.json"
        model_path = self.test_root / "model.pkl"
        rows = [
            "age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target",
            "63,1,3,145,233,1,0,150,0,2.3,0,0,1,1",
            "37,1,2,130,250,0,1,187,0,3.5,0,0,2,1",
            "41,0,1,130,204,0,0,172,0,1.4,2,0,2,1",
            "57,0,0,140,192,0,1,148,0,0.4,1,0,1,0",
        ]
        data_path.write_text("\n".join(rows), encoding="utf-8")

        def fake_train(**kwargs):
            Path(kwargs["model_path"]).write_text("candidate", encoding="utf-8")
            return {"accuracy": 0.85, "roc_auc": 0.9}

        with patch("retrain_if_needed.train", side_effect=fake_train):
            result = retrain_if_needed.retrain_if_needed(
                data_path=data_path,
                model_path=model_path,
                metadata_path=metadata_path,
                target_column="target",
                random_state=42,
                experiment_name="test",
                run_name=None,
                min_rows=4,
                min_accuracy=0.7,
                min_roc_auc=0.7,
                allow_regression=False,
                force=False,
            )

        self.assertEqual(result["status"], "retrained")
        self.assertTrue(model_path.exists())
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["metrics"]["accuracy"], 0.85)


if __name__ == "__main__":
    unittest.main()
