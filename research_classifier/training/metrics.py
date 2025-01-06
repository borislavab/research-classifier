from typing import Dict

import numpy as np
from transformers.trainer_utils import EvalPrediction
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    logits = eval_pred.predictions
    probs = sigmoid(logits)
    y_pred = (probs > 0.5).astype(int)
    y_true = eval_pred.label_ids.astype(int)

    # In multilabel classification, this function computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)

    # When ``true positive + false positive == 0``, precision is undefined - exclude these labels
    labels_with_positive_samples = np.where(y_true.sum(axis=0) > 0)[0]
    precision_micro = precision_score(
        y_true=y_true,
        y_pred=y_pred,
        average="micro",
        labels=labels_with_positive_samples,
        zero_division=0,
    )
    precision_macro = precision_score(
        y_true=y_true,
        y_pred=y_pred,
        average="macro",
        labels=labels_with_positive_samples,
        zero_division=0,
    )
    precision_weighted = precision_score(
        y_true=y_true,
        y_pred=y_pred,
        average="weighted",
        labels=labels_with_positive_samples,
        zero_division=0,
    )
    precision_samples = precision_score(
        y_true=y_true,
        y_pred=y_pred,
        average="samples",
        labels=labels_with_positive_samples,
        zero_division=0,
    )

    # When ``true positive + false negative == 0``, recall is undefined - exclude these labels
    recall_micro = recall_score(
        y_true=y_true,
        y_pred=y_pred,
        average="micro",
        labels=labels_with_positive_samples,
        zero_division=0,
    )
    recall_macro = recall_score(
        y_true=y_true,
        y_pred=y_pred,
        average="macro",
        labels=labels_with_positive_samples,
        zero_division=0,
    )
    recall_weighted = recall_score(
        y_true=y_true,
        y_pred=y_pred,
        average="weighted",
        labels=labels_with_positive_samples,
        zero_division=0,
    )
    recall_samples = recall_score(
        y_true=y_true,
        y_pred=y_pred,
        average="samples",
        labels=labels_with_positive_samples,
        zero_division=0,
    )

    f1_micro = f1_score(
        y_true=y_true,
        y_pred=y_pred,
        average="micro",
        labels=labels_with_positive_samples,
        zero_division=0,
    )
    f1_macro = f1_score(
        y_true=y_true,
        y_pred=y_pred,
        average="macro",
        labels=labels_with_positive_samples,
        zero_division=0,
    )
    f1_weighted = f1_score(
        y_true=y_true,
        y_pred=y_pred,
        average="weighted",
        labels=labels_with_positive_samples,
        zero_division=0,
    )
    f1_samples = f1_score(
        y_true=y_true,
        y_pred=y_pred,
        average="samples",
        labels=labels_with_positive_samples,
        zero_division=0,
    )

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
