"""Train a DistilBERT text classification model using Hugging Face Transformers and datasets.

Usage:
    python src/train_sentiment.py --data data/cleaned/clean_feedback.csv --out_dir models/sentiment

Notes:
 - This script expects 'clean_text' and 'label' columns in the CSV.
 - For a real run, use GPU; adjust TrainingArguments accordingly.
"""
import argparse
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate
import os
import random
import torch

MODEL_NAME = "distilbert-base-uncased"

def prepare_dataset(csv_path, label_list=None, test_size=0.15, seed=42):
    df = pd.read_csv(csv_path)
    if 'clean_text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'clean_text' and 'label' columns.")

    # Build label list if not provided
    if label_list is None:
        label_list = sorted(df['label'].unique().tolist())

    label2id = {l:i for i,l in enumerate(label_list)}
    df['label_id'] = df['label'].map(label2id)
    ds = Dataset.from_pandas(df[['clean_text','label_id']].rename(columns={'clean_text':'text','label_id':'label'}))
    ds = ds.train_test_split(test_size=test_size, seed=seed)
    return ds, label_list

def tokenize_batch(batch, tokenizer, max_length=128):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=max_length)

def compute_metrics(pred):
    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    metric_prec = evaluate.load("precision")
    metric_rec = evaluate.load("recall")
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    acc = metric_acc.compute(predictions=preds, references=labels)
    f1 = metric_f1.compute(predictions=preds, references=labels, average="macro")
    prec = metric_prec.compute(predictions=preds, references=labels, average="macro")
    rec = metric_rec.compute(predictions=preds, references=labels, average="macro")
    return {
        "accuracy": float(acc["accuracy"]),
        "f1_macro": float(f1["f1"]),
        "precision_macro": float(prec["precision"]),
        "recall_macro": float(rec["recall"])
    }

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(data_csv, out_dir, epochs=3, batch_size=16, lr=2e-5):
    set_seed(42)
    ds, label_list = prepare_dataset(data_csv)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenized = ds.map(lambda x: tokenize_batch(x, tokenizer), batched=True)
    tokenized = tokenized.remove_columns([c for c in tokenized['train'].column_names if c not in ['input_ids','attention_mask','label']])
    tokenized.set_format(type='torch', columns=['input_ids','attention_mask','label'])

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(label_list))

    device = 0 if torch.cuda.is_available() else -1

    training_args = TrainingArguments(
        output_dir=out_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size*2,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        save_total_limit=2,
        logging_dir=os.path.join(out_dir, "logs"),
        push_to_hub=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['test'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    # Save model + tokenizer
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Model and tokenizer saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="cleaned CSV with clean_text and label columns")
    parser.add_argument("--out_dir", default="models/sentiment")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()
    main(args.data, args.out_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
