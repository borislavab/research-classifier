from typing import Dict

import numpy as np
from datasets import EvalPrediction
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    logits = eval_pred.predictions
    print(logits.shape)
    print(logits)
    probs = sigmoid(logits)
    y_pred = (probs > 0.5).astype(int)
    y_true = eval_pred.label_ids.astype(int)

    # In multilabel classification, this function computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)

    precision_micro = precision_score(y_true=y_true, y_pred=y_pred, average="micro")
    precision_macro = precision_score(y_true=y_true, y_pred=y_pred, average="macro")
    precision_weighted = precision_score(
        y_true=y_true, y_pred=y_pred, average="weighted"
    )
    precision_samples = precision_score(y_true=y_true, y_pred=y_pred, average="samples")

    recall_micro = recall_score(y_true=y_true, y_pred=y_pred, average="micro")
    recall_macro = recall_score(y_true=y_true, y_pred=y_pred, average="macro")
    recall_weighted = recall_score(y_true=y_true, y_pred=y_pred, average="weighted")
    recall_samples = recall_score(y_true=y_true, y_pred=y_pred, average="samples")

    f1_micro = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
    f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
    f1_weighted = f1_score(y_true=y_true, y_pred=y_pred, average="weighted")
    f1_samples = f1_score(y_true=y_true, y_pred=y_pred, average="samples")

    return {
        "accuracy": accuracy,
        "precision_micro": precision_micro,
        "precision_macro": precision_macro,
        "precision_weighted": precision_weighted,
        "precision_samples": precision_samples,
        "recall_micro": recall_micro,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,
        "recall_samples": recall_samples,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f1_samples": f1_samples,
    }
