from django.conf import settings
from .classifier import ArticleClassifier
import logging
from typing import List
from celery import shared_task
from celery.signals import worker_ready

logger = logging.getLogger(__name__)

classifier = None


@worker_ready.connect
def initialize_classifier(sender=None, **kwargs):
    global classifier
    logger.info("Initializing classifier in worker process")
    classifier = ArticleClassifier(settings.MODEL_CHECKPOINT_PATH)


@shared_task
def predict_article(article: str) -> List[str]:
    global classifier
    if classifier is None:
        # Fallback initialization
        logger.warning("Classifier not initialized, initializing now")
        initialize_classifier()

    logger.info("Starting prediction task")
    try:
        predictions = classifier.predict(article)
        logger.info(f"Prediction successful: {predictions}")
        return predictions
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise e
