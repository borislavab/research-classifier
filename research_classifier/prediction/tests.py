from django.test import TestCase, Client
import json


class PredictionEndpointTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_predict_endpoint(self):
        # Test valid input
        response = self.client.post(
            "/api/predict/",
            data=json.dumps(
                {"article": "This is a test article about machine learning."}
            ),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertIn("predictions", data)
        self.assertEqual(data["status"], "success")

    def test_predict_endpoint_invalid_input(self):
        # Test missing article
        response = self.client.post(
            "/api/predict/", data=json.dumps({}), content_type="application/json"
        )

        self.assertEqual(response.status_code, 400)

        # Test invalid JSON
        response = self.client.post(
            "/api/predict/", data="invalid json", content_type="application/json"
        )

        self.assertEqual(response.status_code, 400)
