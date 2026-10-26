#!/usr/bin/env python3
"""Classify one held-out test row or one new question/triple with the saved model."""

import argparse
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(HERE / ".cache" / "huggingface"))

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import load_saved_splits
from modeling import model_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-index", type=int, help="Zero-based position among the held-out test rows; defaults to 0")
    parser.add_argument("--question", help="Competency question for a new record")
    parser.add_argument("--subject", help="Triple subject for a new record")
    parser.add_argument("--relation", help="Triple relation for a new record")
    parser.add_argument("--object", help="Triple object for a new record")
    parser.add_argument("--gold-label", type=int, choices=(0, 1), help="Optional known label for a new record")
    parser.add_argument("--output-dir", type=Path, default=HERE / "artifacts")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
    fields = ("question", "subject", "relation", "object")
    supplied = [getattr(args, field) is not None for field in fields]
    if any(supplied):
        if args.test_index is not None or not all(supplied):
            parser.error("Provide all four text fields together, without --test-index")
        row = {"competency_question": args.question, "subject": args.subject,
               "relation": args.relation, "object": args.object}
        if any(not row[key].strip() for key in row):
            parser.error("Question, subject, relation, and object must be nonempty")
        source = "new_record"
        gold = args.gold_label
    else:
        if args.gold_label is not None:
            parser.error("--gold-label is only for a new record")
        index = 0 if args.test_index is None else args.test_index
        rows, assignments = load_saved_splits(config, output_dir)
        test_rows = [row for row in rows if assignments[row["row_id"]] == "test"]
        if not 0 <= index < len(test_rows):
            parser.error(f"--test-index must be between 0 and {len(test_rows) - 1}")
        row = test_rows[index]
        source = "held_out_test"
        gold = row["gold_label"]

    model_dir = output_dir / "model"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, fix_mistral_regex=False, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    question, triple = model_inputs(row)
    inputs = tokenizer(question, triple, return_tensors="pt", truncation=True,
                       max_length=config["max_length"])
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        probabilities = torch.softmax(model(**inputs).logits.float(), dim=-1)[0].cpu().tolist()
    prediction = int(probabilities[1] >= 0.5)
    result = {
        "source": source,
        "row_id": row.get("row_id"),
        "question": question,
        "subject": row["subject"],
        "relation": row["relation"],
        "object": row["object"],
        "predicted_label": prediction,
        "probability_class_0": probabilities[0],
        "probability_class_1": probabilities[1],
        "device": str(device),
    }
    if gold is not None:
        result["gold_label"] = gold
        result["correct"] = prediction == gold
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
