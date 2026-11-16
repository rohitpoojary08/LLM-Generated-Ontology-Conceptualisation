#!/usr/bin/env python3
"""Evaluate the saved best model on its held-out test split."""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent / ".cache" / "huggingface"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from data import HERE, load_saved_splits
from modeling import CriticDataset, predict, seed_everything


def evaluate_model(config, output_dir, split="test"):
    rows, assignments = load_saved_splits(config, output_dir)
    selected = [row for row in rows if assignments[row["row_id"]] == split]
    if not selected:
        raise ValueError(f"No rows in {split} split")
    model_dir = output_dir / "model"
    # Transformers 4.57 misidentifies locally saved DeBERTa tokenizers as Mistral.
    tokenizer = AutoTokenizer.from_pretrained(model_dir, fix_mistral_regex=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(
        CriticDataset(selected, tokenizer, config["max_length"]),
        batch_size=config["batch_size"], shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer),
    )
    gold, predicted, positive_probability = predict(model, loader, device)
    report = classification_report(gold, predicted, labels=[0, 1], target_names=["class_0", "class_1"], output_dict=True, zero_division=0)
    matrix = confusion_matrix(gold, predicted, labels=[0, 1])
    metrics = {
        "split": split, "count": len(selected), "device": str(device),
        "accuracy": report["accuracy"],
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"],
        "f1_macro": report["macro avg"]["f1-score"],
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": [0, 1],
    }
    prefix = "test" if split == "test" else split
    # Reuse held-out predictions to report performance for each data source.
    source_names = {"WDV-CQ-HA": "WDV-CQ", "CQ2Term": "CQ2Term", "CQ2Onto": "CQ2Onto"}
    source_metrics = {}
    for source in sorted({row["source"] for row in selected}):
        indices = [i for i, row in enumerate(selected) if row["source"] == source]
        source_gold = [gold[i] for i in indices]
        source_predicted = [predicted[i] for i in indices]
        precision, recall, f1, _ = precision_recall_fscore_support(
            source_gold, source_predicted, labels=[0, 1], average="macro", zero_division=0,
        )
        source_metrics[source_names.get(source, source)] = {
            "source": source,
            "split": split,
            "count": len(indices),
            "accuracy": float(accuracy_score(source_gold, source_predicted)),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1),
            "classification_report": classification_report(
                source_gold, source_predicted, labels=[0, 1],
                target_names=["class_0", "class_1"], output_dict=True, zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                source_gold, source_predicted, labels=[0, 1],
            ).tolist(),
            "confusion_matrix_labels": [0, 1],
        }
    (output_dir / f"{prefix}_metrics_by_source.json").write_text(
        json.dumps(source_metrics, indent=2), encoding="utf-8",
    )
    columns = ["source", "count", "accuracy", "precision_macro", "recall_macro", "f1_macro"]
    with (output_dir / f"{prefix}_metrics_by_source.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for name, values in source_metrics.items():
            writer.writerow({"source": name, **{column: values[column] for column in columns[1:]}})
    (output_dir / f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output_dir / f"{prefix}_classification_report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "precision", "recall", "f1_score", "support"])
        for key in ("class_0", "class_1", "macro avg", "weighted avg"):
            item = report[key]
            writer.writerow([key, item["precision"], item["recall"], item["f1-score"], item["support"]])
        writer.writerow(["accuracy", "", "", report["accuracy"], len(selected)])
    with (output_dir / f"{prefix}_confusion_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", "0", "1"])
        for label, values in enumerate(matrix):
            writer.writerow([label, *values])
    with (output_dir / f"{prefix}_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "gold_label", "predicted_label", "positive_probability"])
        for row, true, pred, prob in zip(selected, gold, predicted, positive_probability):
            writer.writerow([row["row_id"], true, pred, prob])
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["0", "1"], yticklabels=["0", "1"],
             xlabel="Predicted label", ylabel="True label", title=f"{split.title()} confusion matrix")
    for i in (0, 1):
        for j in (0, 1):
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    figure.tight_layout()
    figure.savefig(output_dir / f"{prefix}_confusion_matrix.png", dpi=180)
    plt.close(figure)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--output-dir", type=Path, default=HERE / "artifacts")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed_everything(config["seed"])
    metrics = evaluate_model(config, args.output_dir.resolve())
    print(json.dumps({key: metrics[key] for key in ("count", "device", "accuracy", "precision_macro", "recall_macro", "f1_macro", "confusion_matrix")}, indent=2))


if __name__ == "__main__":
    main()
