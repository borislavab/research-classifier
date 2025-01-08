import pytest
from django.urls import reverse
import json
from unittest.mock import Mock
from celery.result import states


@pytest.fixture
def predict_url():
    return reverse("predict")


@pytest.fixture
def article_text():
    return "Writing a multi-label research classifier with BERT is very science-y"


class TestPredictionAPI:
    def test_predict_endpoint(self, client, predict_url, article_text, mocker):
        # Setup mock
        mock_task = Mock()
        mock_task.id = "test-task-id"
        mock_predict = mocker.patch(
            "research_classifier.prediction.views.predict_article"
        )
        mock_predict.delay.return_value = mock_task

        # Make request
        response = client.post(
            predict_url,
            data=json.dumps({"article": article_text}),
            content_type="application/json",
        )

        # Assert response
        assert response.status_code == 202
        assert response["Location"] == f"/api/prediction/{mock_task.id}"
        assert response["Retry-After"] == "1"

        data = response.json()
        assert data["task_id"] == "test-task-id"
        assert data["status"] == "pending"
        assert "created_at" in data

        # Verify task was called
        mock_predict.delay.assert_called_once_with(article_text)

    def test_get_prediction_success(self, client, mocker):
        # Setup mock
        task_id = "test-task-id"
        mock_result = Mock()
        mock_result.status = states.SUCCESS
        mock_result.result = ["cs.AI", "cs.LG"]

        mock_async_result = mocker.patch(
            "research_classifier.prediction.views.AsyncResult"
        )
        mock_async_result.return_value = mock_result

        # Make request
        response = client.get(reverse("get_prediction", args=[task_id]))

        # Assert response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["predictions"] == ["cs.AI", "cs.LG"]

    def test_get_prediction_pending(self, client, mocker):
        # Setup mock
        task_id = "test-task-id"
        mock_result = Mock()
        mock_result.status = states.PENDING
        mock_result.result = None

        mock_async_result = mocker.patch(
            "research_classifier.prediction.views.AsyncResult"
        )
        mock_async_result.return_value = mock_result

        # Make request
        response = client.get(reverse("get_prediction", args=[task_id]))

        # Assert response
        assert response.status_code == 202
        assert response["Retry-After"] == "1"
        data = response.json()
        assert data["status"] == "pending"

    def test_predict_missing_article(self, client, predict_url):
        response = client.post(
            predict_url, data=json.dumps({}), content_type="application/json"
        )
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "article" in data["error"]
