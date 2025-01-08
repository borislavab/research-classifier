from django.conf import settings
from .classifier import ArticleClassifier
import logging
from typing import List
from celery import shared_task

logger = logging.getLogger(__name__)

classifier = ArticleClassifier(settings.MODEL_CHECKPOINT_PATH)


@shared_task
def predict_article(article: str) -> List[str]:
    logger.info("Starting prediction task")
    try:
        predictions = classifier.predict(article)
        logger.info(f"Prediction successful: {predictions}")
        return predictions
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise e
