from celery import shared_task
from .classifier import ArticleClassifier
import logging

logger = logging.getLogger(__name__)

classifier = ArticleClassifier("research_classifier/prediction/model/checkpoint-3000")


@shared_task
def predict_article(article):
    logger.info("Starting prediction task")
    try:
        predictions = classifier.predict(article)
        logger.info(f"Prediction successful: {predictions}")
        return {"predictions": predictions, "status": "success"}
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise
