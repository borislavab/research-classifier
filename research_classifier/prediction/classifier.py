from transformers import BertForSequenceClassification
from typing import List
import torch
from research_classifier.preprocessing import Pipeline, DefaultTokenizer


class ArticleClassifier:
    def __init__(self, model_path: str):
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        # TODO: store pipeline settings along with model so preprocessing can easily change
        # but remain identical in training and prediction
        # TODO: store tokenizer along with model so upstream changes don't affect model prediction
        self.pipeline = Pipeline(DefaultTokenizer())
        self.labels = self.pipeline.get_labels()

    def predict(self, article: str) -> List[str]:
        tokenized = self.pipeline.process_abstract(article)

        with torch.no_grad():
            outputs = self.model(**tokenized)
            logits = outputs.logits
            probabilities = torch.sigmoid(logits).squeeze()

        label_indices = torch.where(probabilities > 0.5)[0]
        labels = [self.labels[int(i)] for i in label_indices]
        return labels


if __name__ == "__main__":
    classifier = ArticleClassifier("./prediction/model/checkpoint-3000")
    print(
        classifier.predict(
            "This paper presents a new method for image segmentation that uses a convolutional neural network (CNN) to predict the segmentation of an image. The method is based on the U-Net architecture, which is a popular architecture for image segmentation tasks. The method is evaluated on the Pascal VOC 2012 dataset, which is a widely used dataset for image segmentation tasks. The method achieves state-of-the-art performance on the dataset."
        )
    )
