import unittest

from main import ModelInput, health_check, predict


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
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
        body = health_check()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["model_path"], "model.pkl")

    def test_predict_returns_prediction_and_label(self) -> None:
        body = predict(ModelInput(**self.valid_payload))
        self.assertIn(body["prediction"], [0, 1])
        self.assertIn(body["risk_label"], ["low_risk", "high_risk"])
        self.assertGreaterEqual(body["probability"], 0)
        self.assertLessEqual(body["probability"], 1)

    def test_predict_rejects_invalid_input(self) -> None:
        invalid_payload = dict(self.valid_payload)
        invalid_payload["sex"] = 3

        with self.assertRaises(Exception):
            ModelInput(**invalid_payload)


if __name__ == "__main__":
    unittest.main()
