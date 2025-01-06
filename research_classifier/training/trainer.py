from transformers import (
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from research_classifier.preprocessing.categories import get_labels
from research_classifier.training.dataset import collator, load_for_training
from research_classifier.training.metrics import compute_metrics
import json
import sys
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_model():
    # labels count is 158
    labels = get_labels()
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}
    print("Label count is ", len(labels))

    return BertForSequenceClassification.from_pretrained(
        "bert-base-cased",
        num_labels=len(labels),
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
    )


def get_trainer(
    dataset_path: str = None,
    output_dir: str = "./output",
    num_epochs: int = 6,
    sample_head: int = None,
):
    (train_dataset, eval_dataset) = load_for_training(
        dataset_path=dataset_path, head=sample_head
    )
    print(f"Will train for {num_epochs} epochs")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    print(f"Device: {device}")

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=500,
        # increase batch size since GPU is underutilized
        per_device_train_batch_size=1 if device == "cpu" else 64,
        per_device_eval_batch_size=1 if device == "cpu" else 64,
        num_train_epochs=num_epochs,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to="none",
        # Enable mixed precision for faster computation
        fp16=(device == "cuda"),
    )

    trainer = Trainer(
        model=get_model(),
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=collator,
    )

    return trainer


def train_and_save(trainer):
    trainer.train()
    trainer.save_model("./output_final/model")

    # Save metrics
    metrics = trainer.evaluate()
    with open("./output_final/metrics.json", "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else None
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./output"
    num_epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    sample_count = int(sys.argv[4]) if len(sys.argv) > 4 else None
    trainer = get_trainer(dataset_path, output_dir, num_epochs, sample_count)
    train_and_save(trainer)
