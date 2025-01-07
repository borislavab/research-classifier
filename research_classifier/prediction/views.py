from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
from .tasks import predict_article
import logging
from celery.result import AsyncResult

logger = logging.getLogger(__name__)


@csrf_exempt
@require_http_methods(["POST"])
def predict(request):
    try:
        data = json.loads(request.body)

        if "article" not in data:
            return JsonResponse(
                {"error": "Missing 'article' field in request body"}, status=400
            )
        article = data["article"]

        # Submit task to Celery for asynchronous processing outside the web server process
        task = predict_article.delay(article)
        logger.info(f"Task submitted with ID: {task.id}")

        # Return task ID immediately
        return JsonResponse({"task_id": task.id, "status": "processing"})

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON in request body"}, status=400)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def get_prediction(request, task_id):
    logger.info(f"Getting prediction for task ID {task_id}")
    task = AsyncResult(task_id)
    logger.info(f"Response: {task.status}")
    status = task.status
    traceback = task.traceback
    result = task.result
    if isinstance(result, Exception):
        return JsonResponse(
            {"status": status, "error": str(result), "traceback": traceback}
        )
    if task.status == "PENDING":
        return JsonResponse({"status": "pending"})
    if task.status == "SUCCESS":
        return JsonResponse(task.result)
    return JsonResponse({"status": "unknown"})
