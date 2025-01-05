from transformers import (
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from preprocessing.dataset import LABELS
from training.dataset import collator, load_for_training
from training.metrics import compute_metrics
import json
import sys


def get_model():
    # categories count is 176
    categories_count = len(LABELS)
    id2label = {idx: label for idx, label in enumerate(LABELS)}
    label2id = {label: idx for idx, label in enumerate(LABELS)}

    return BertForSequenceClassification.from_pretrained(
        "bert-base-cased",
        num_labels=categories_count,
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
    )


def get_trainer(dataset_path: str = None, output_dir: str = "./output"):
    (train_dataset, eval_dataset) = load_for_training(dataset_path)

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        save_strategy="epoch",
        save_total_limit=2,
        evaluation_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=8,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
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
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    trainer = get_trainer(dataset_path, output_dir)
    train_and_save(trainer)
