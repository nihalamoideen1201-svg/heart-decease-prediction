import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import main


class FakeModelWithProbability:
    def predict(self, frame):
        return [1]

    def predict_proba(self, frame):
        return [[0.12, 0.88]]


class FakeModelWithoutProbability:
    def predict(self, frame):
        return [0]


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(main.app)
        self.valid_payload = {
            "age": 52,
            "sex": 1,
            "cp": 0,
            "trestbps": 125,
            "chol": 212,
            "fbs": 0,
            "restecg": 1,
            "thalach": 168,
            "exang": 0,
            "oldpeak": 1.2,
            "slope": 1,
            "ca": 0,
            "thal": 2,
        }

    def test_health_endpoint_is_ok(self) -> None:
        with patch("main.load_model", return_value=FakeModelWithProbability()):
            response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "model_path": Path(main.MODEL_PATH).name})

    def test_health_endpoint_returns_503_when_model_load_fails(self) -> None:
        with patch("main.load_model", side_effect=RuntimeError("load failed")):
            response = self.client.get("/health")

        self.assertEqual(response.status_code, 503)
        self.assertIn("Model unavailable", response.json()["detail"])

    def test_predict_returns_prediction_label_and_probability(self) -> None:
        with patch("main.load_model", return_value=FakeModelWithProbability()):
            response = self.client.post("/predict", json=self.valid_payload)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["prediction"], 1)
        self.assertEqual(body["risk_label"], "high_risk")
        self.assertEqual(body["probability"], 0.88)

    def test_predict_omits_probability_when_model_does_not_support_it(self) -> None:
        with patch("main.load_model", return_value=FakeModelWithoutProbability()):
            response = self.client.post("/predict", json=self.valid_payload)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["prediction"], 0)
        self.assertEqual(body["risk_label"], "low_risk")
        self.assertNotIn("probability", body)

    def test_predict_returns_500_when_model_file_is_missing(self) -> None:
        main.load_model.cache_clear()
        missing_model_path = main.APP_DIR / "missing-model.pkl"
        with patch.object(main, "MODEL_PATH", missing_model_path):
            response = self.client.post("/predict", json=self.valid_payload)
        main.load_model.cache_clear()

        self.assertEqual(response.status_code, 500)
        self.assertIn(str(missing_model_path), response.json()["detail"])

    def test_predict_rejects_out_of_range_values(self) -> None:
        invalid_payload = dict(self.valid_payload)
        invalid_payload["sex"] = 3

        response = self.client.post("/predict", json=invalid_payload)

        self.assertEqual(response.status_code, 422)
        self.assertIn("less than or equal to 1", response.text)

    def test_predict_rejects_missing_required_field(self) -> None:
        invalid_payload = dict(self.valid_payload)
        del invalid_payload["thal"]

        response = self.client.post("/predict", json=invalid_payload)

        self.assertEqual(response.status_code, 422)
        self.assertIn("Field required", response.text)

    def test_predict_rejects_extra_fields(self) -> None:
        invalid_payload = dict(self.valid_payload)
        invalid_payload["unexpected"] = 99

        response = self.client.post("/predict", json=invalid_payload)

        self.assertEqual(response.status_code, 422)
        self.assertIn("Extra inputs are not permitted", response.text)


if __name__ == "__main__":
    unittest.main()
