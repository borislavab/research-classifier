from typing import Dict

import numpy as np
from transformers.trainer_utils import EvalPrediction
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss

import evaluate
import torch

f1_metric = evaluate.load("f1")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# WIP, not used yet, this would be used if eval_batch_metrics is set to True in Trainer
# in order to speed up evaluation step
# however I understood that the evaluation was slow because the evaluation set was too big so this was not needed
def compute_metrics_batched(
    eval_pred: EvalPrediction, compute_result: bool = True
) -> Dict[str, float]:
    logits = eval_pred.predictions
    print(type(logits))
    print(logits.shape)
    probs = torch.sigmoid(logits)
    y_pred = torch.where(probs > 0.5, 1, 0)
    print(logits.shape)
    y_true = eval_pred.label_ids.int()

    print(y_pred.shape)
    print(y_true.shape)
    for preds, refs in zip(y_pred, y_true):
        print(preds.shape)
        print(refs.shape)
        f1_metric.add_batch(references=refs, predictions=preds)
    print(compute_result)
    if compute_result:
        return f1_metric.compute(average="macro")
    return {}


def compute_metrics_debug_labels(eval_pred: EvalPrediction) -> Dict[str, float]:
    logits = eval_pred.predictions
    probs = sigmoid(logits)
    y_pred = (probs > 0.5).astype(int)
    y_true = eval_pred.label_ids.astype(int)

    labels_with_positive_samples = np.where(y_true.sum(axis=0) > 0)[0]
    print(labels_with_positive_samples)
    precision_per_label = precision_score(
        y_true=y_true,
        y_pred=y_pred,
        average=None,
        zero_division=0,
    )
    recall_per_label = recall_score(
        y_true=y_true,
        y_pred=y_pred,
        average=None,
        zero_division=0,
    )
    f1_per_label = f1_score(
        y_true=y_true,
        y_pred=y_pred,
        average=None,
        zero_division=0,
    )

    results = {}
    for label in range(y_true.shape[1]):
        results[f"precision_{label}"] = precision_per_label[label]
        results[f"recall_{label}"] = recall_per_label[label]
        results[f"f1_{label}"] = f1_per_label[label]

    return results


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    logits = eval_pred.predictions
    probs = sigmoid(logits)
    y_pred = (probs > 0.5).astype(int)
    y_true = eval_pred.label_ids.astype(int)

    # In multilabel classification, this function computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)

    # Hamming loss is better for multi-label classification problems as it accounts for label-level correct predictions
    # Smaller hamming loss is better
    hamming = hamming_loss(y_true=y_true, y_pred=y_pred)

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

    # Include all metrics to track which one improves/suffers most to tweak training
    # In practice, hamming loss, f1_macro, and f1_weighted are the most telling indicators of overall performance
    return {
        "accuracy": accuracy,
        "hamming_loss": hamming,
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
