from transformers import (
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from preprocessing.dataset import LABELS
from training.dataset import collator, load_for_training
from training.metrics import compute_metrics

(train_dataset, eval_dataset) = load_for_training()

# categories count is 176
categories_count = len(LABELS)
id2label = {idx: label for idx, label in enumerate(LABELS)}
label2id = {label: idx for idx, label in enumerate(LABELS)}

model = BertForSequenceClassification.from_pretrained(
    "bert-base-cased",
    num_labels=categories_count,
    problem_type="multi_label_classification",
    id2label=id2label,
    label2id=label2id,
)

training_arguments = TrainingArguments(
    output_dir=".",
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=8,
)

trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=collator,
)

trainer.train()
