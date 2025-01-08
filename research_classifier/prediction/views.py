from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .tasks import predict_article
import logging
from celery.result import AsyncResult
from datetime import datetime
from celery import states
from .dtos import OperationStatus

logger = logging.getLogger(__name__)


@api_view(["POST"])
def predict(request):
    try:
        data = request.data
        logger.info(f"Prediction request received: {data}")

        if "article" not in data:
            return Response(
                {"error": "Missing 'article' field in request body"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        article = data["article"]

        # Submit task to Celery for asynchronous processing outside the web server process
        task = predict_article.delay(article)
        logger.info(f"Task submitted with ID: {task.id}")

        # Return task ID immediately
        return Response(
            {
                "task_id": task.id,
                "status": OperationStatus.PENDING,
                "created_at": datetime.now().isoformat(),
            },
            status=status.HTTP_202_ACCEPTED,
            headers={"Location": f"/api/prediction/{task.id}", "Retry-After": "1"},
        )
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(["GET"])
def get_prediction(request, task_id):
    logger.info(f"Getting prediction for task ID {task_id}")
    task: AsyncResult = AsyncResult(task_id)
    logger.info(f"Response: {task.status}")
    if isinstance(task.result, Exception) or task.status == states.FAILURE:
        return Response(
            {
                "status": OperationStatus.ERROR,
                "error": str(task.result),
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    match task.status:
        case states.SUCCESS:
            return Response(
                {
                    "predictions": task.result,
                    "status": OperationStatus.SUCCESS,
                },
                status=status.HTTP_200_OK,
            )
        case _:
            return Response(
                {
                    "status": (
                        OperationStatus.PENDING
                        if task.status == states.PENDING
                        else OperationStatus.PROCESSING
                    ),
                },
                status=status.HTTP_202_ACCEPTED,
                headers={"Retry-After": "1"},
            )
