from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
from research_classifier.prediction.classifier import ArticleClassifier

classifier = ArticleClassifier("research_classifier/prediction/model/checkpoint-3000")


@csrf_exempt
@require_http_methods(["POST"])
def predict(request):
    try:
        # Parse JSON body
        data = json.loads(request.body)

        # Validate input
        if "article" not in data:
            return JsonResponse(
                {"error": "Missing 'article' field in request body"}, status=400
            )

        # Get prediction
        article = data["article"]
        predictions = classifier.predict(article)

        # Return response
        return JsonResponse({"predictions": predictions, "status": "success"})

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON in request body"}, status=400)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
