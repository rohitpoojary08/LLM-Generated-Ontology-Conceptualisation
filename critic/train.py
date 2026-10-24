#!/usr/bin/env python3
"""Fine-tune DeBERTa-v3-small and evaluate the best validation-F1 checkpoint."""

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent / ".cache" / "huggingface"))

import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup

from data import HERE, prepare
from modeling import CriticDataset, predict, seed_everything


def train(config, output_dir):
    seed_everything(config["seed"])
    rows, assignments = prepare(config, output_dir)
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"], num_labels=2,
        id2label={0: "class_0", 1: "class_1"},
        label2id={"class_0": 0, "class_1": 1},
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    collator = DataCollatorWithPadding(tokenizer)
    train_rows = [row for row in rows if assignments[row["row_id"]] == "train"]
    val_rows = [row for row in rows if assignments[row["row_id"]] == "validation"]
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(CriticDataset(train_rows, tokenizer, config["max_length"]),
                              batch_size=config["batch_size"], shuffle=True, generator=generator, collate_fn=collator)
    val_loader = DataLoader(CriticDataset(val_rows, tokenizer, config["max_length"]),
                            batch_size=config["batch_size"], shuffle=False, collate_fn=collator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    total_steps = len(train_loader) * config["max_epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(total_steps * config["warmup_fraction"]), total_steps)
    best_f1, best_epoch, stale = -1.0, 0, 0
    logs = []
    model_dir = output_dir / "model"
    for epoch in range(1, config["max_epochs"] + 1):
        start = time.monotonic()
        model.train()
        loss_sum = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item()
        gold, predicted, _ = predict(model, val_loader, device)
        val_f1 = f1_score(gold, predicted, average="macro", zero_division=0)
        log = {"epoch": epoch, "train_loss": loss_sum / len(train_loader),
               "validation_f1_macro": val_f1, "seconds": time.monotonic() - start}
        logs.append(log)
        print(json.dumps(log), flush=True)
        with (output_dir / "training_log.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(log))
            writer.writeheader()
            writer.writerows(logs)
        if val_f1 > best_f1:
            best_f1, best_epoch, stale = val_f1, epoch, 0
            model.save_pretrained(model_dir, safe_serialization=True)
            tokenizer.save_pretrained(model_dir)
            (output_dir / "best_checkpoint.json").write_text(
                json.dumps({"epoch": epoch, "validation_f1_macro": val_f1, "model_dir": str(model_dir)}, indent=2),
                encoding="utf-8",
            )
        else:
            stale += 1
            if stale >= config["early_stopping_patience"]:
                print(f"Early stopping after epoch {epoch}; best epoch {best_epoch}", flush=True)
                break
    return best_epoch, best_f1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--output-dir", type=Path, default=HERE / "artifacts")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    best_epoch, best_f1 = train(config, output_dir)
    from evaluate import evaluate_model
    metrics = evaluate_model(config, output_dir, "test")
    print(json.dumps({"best_epoch": best_epoch, "best_validation_f1_macro": best_f1,
                      "test_accuracy": metrics["accuracy"], "test_f1_macro": metrics["f1_macro"]}, indent=2))


if __name__ == "__main__":
    main()
