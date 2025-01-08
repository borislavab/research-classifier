from transformers import (
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from research_classifier.preprocessing.categories import get_labels
from research_classifier.training.dataset import collator, load_for_training
from research_classifier.training.metrics import (
    compute_metrics,
    compute_metrics_debug_labels,
)
import json
import sys
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_model(model_path: str = None):
    # labels count is 158
    labels = get_labels()
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}
    print("Label count is ", len(labels))

    # By default, the "multi_label_classification" problem uses
    # sigmoid activation function and binary cross entropy loss
    # To account for class imbalance, we could override the loss function
    # weighted binary cross entropy or focal loss
    return BertForSequenceClassification.from_pretrained(
        model_path or "bert-base-cased",
        num_labels=len(labels),
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
    )


def get_training_args(output_dir: str, num_epochs: int):
    return TrainingArguments(
        output_dir=output_dir,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=500,
        # increase batch size since GPU is underutilized
        per_device_train_batch_size=1 if device == "cpu" else 64,
        per_device_eval_batch_size=2 if device == "cpu" else 64,
        num_train_epochs=num_epochs,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to="none",
        # Enable mixed precision for faster computation
        fp16=(device == "cuda"),
        # TODO: currently evaluation seems to be slow, might be due to an OOM issue observed
        # evaluating in batches could resolve this, requires refactoring of compute_metrics - WIP
        # batch_eval_metrics=True,
    )


def get_trainer(
    dataset_path: str = None,
    output_dir: str = "./output",
    num_epochs: int = 6,
    sample_head: int = None,
    model_path: str = None,
    custom_compute_metrics=None,
):
    (train_dataset, eval_dataset) = load_for_training(
        dataset_path=dataset_path, head=sample_head
    )
    print(f"Will train for {num_epochs} epochs")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    print(f"Device: {device}")

    trainer = Trainer(
        model=get_model(model_path),
        args=get_training_args(output_dir, num_epochs),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # use compute_metrics_batched if batch_eval_metrics is set to True
        compute_metrics=custom_compute_metrics or compute_metrics,
        data_collator=collator,
    )

    return trainer


def evaluate(
    model_path: str,
    output_dir: str = "./output",
    dataset_path: str = None,
    sample_head: int = None,
):
    trainer = get_trainer(
        model_path=model_path,
        output_dir=output_dir,
        dataset_path=dataset_path,
        sample_head=sample_head,
        custom_compute_metrics=compute_metrics_debug_labels,
    )
    metrics = trainer.evaluate()
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f)


def train_and_save(trainer: Trainer, from_scratch: bool = False):
    trainer.train(resume_from_checkpoint=not from_scratch)


if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else None
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./output"
    num_epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    sample_count = int(sys.argv[4]) if len(sys.argv) > 4 else None
    trainer = get_trainer(dataset_path, output_dir, num_epochs, sample_count)
    train_and_save(trainer)
