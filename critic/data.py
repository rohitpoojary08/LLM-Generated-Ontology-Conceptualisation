"""Load and split the critic data without crossing question/triple groups."""

import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REQUIRED = (
    "competency_question", "subject", "relation", "object", "gold_label",
    "original_triple", "source_triple", "cq_origin_triple",
)
GROUP_COLUMNS = ("competency_question", "original_triple", "source_triple", "cq_origin_triple")


def normalize(value):
    return " ".join(value.casefold().split())


def load_rows(dataset_path):
    path = Path(dataset_path).resolve()
    with path.open("rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(REQUIRED) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Dataset is empty")
    for index, row in enumerate(rows):
        if row["gold_label"] not in {"0", "1"}:
            raise ValueError(f"Invalid gold_label on CSV row {index + 2}")
        if any(not row[column].strip() for column in REQUIRED):
            raise ValueError(f"Empty required field on CSV row {index + 2}")
        row["row_id"] = index
        row["gold_label"] = int(row["gold_label"])
    return rows, digest


def make_groups(rows):
    """Union rows sharing a question, provenance triple, or exact S/R/O triple."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        parent[find(a)] = find(b)

    seen = {}
    for index, row in enumerate(rows):
        keys = [(column, normalize(row[column])) for column in GROUP_COLUMNS]
        keys.append(("sro", tuple(normalize(row[column]) for column in ("subject", "relation", "object"))))
        for key in keys:
            if key in seen:
                union(index, seen[key])
            else:
                seen[key] = index
    groups = defaultdict(list)
    for index in range(len(rows)):
        groups[find(index)].append(index)
    return list(groups.values())


def stratified_group_split(rows, groups, config):
    """Assign intact groups near requested row and class counts deterministically."""
    fractions = [config["train_fraction"], config["validation_fraction"], config["test_fraction"]]
    if abs(sum(fractions) - 1.0) > 1e-9 or any(f <= 0 for f in fractions):
        raise ValueError("Split fractions must be positive and sum to 1")
    rng = random.Random(config["seed"])
    order = list(groups)
    rng.shuffle(order)
    # Large or label-imbalanced components go first; shuffle breaks ties reproducibly.
    order.sort(key=lambda group: (len(group), abs(sum(rows[i]["gold_label"] for i in group) / len(group) - 0.5)), reverse=True)
    names = ("train", "validation", "test")
    totals = {name: [0, 0] for name in names}
    assignments = {}
    global_labels = [sum(row["gold_label"] == label for row in rows) for label in (0, 1)]
    target_rows = [len(rows) * fraction for fraction in fractions]
    target_labels = [[count * fraction for count in global_labels] for fraction in fractions]
    for group in order:
        labels = [sum(rows[i]["gold_label"] == label for i in group) for label in (0, 1)]
        choices = []
        for split_index, name in enumerate(names):
            new = [totals[name][j] + labels[j] for j in (0, 1)]
            size = sum(new)
            old_size = sum(totals[name])
            # Compare the change in the overall squared row and class error.
            score = (size - target_rows[split_index]) ** 2 - (old_size - target_rows[split_index]) ** 2
            score += sum(
                (new[j] - target_labels[split_index][j]) ** 2
                - (totals[name][j] - target_labels[split_index][j]) ** 2
                for j in (0, 1)
            )
            choices.append((score, totals[name][0] + totals[name][1], split_index))
        chosen = names[min(choices)[2]]
        for j in (0, 1):
            totals[chosen][j] += labels[j]
        for i in group:
            assignments[i] = chosen
    if set(assignments) != set(range(len(rows))) or any(sum(totals[name]) == 0 for name in names):
        raise ValueError("Could not create three nonempty grouped splits")
    if any(totals[name][j] == 0 for name in names for j in (0, 1)):
        raise ValueError("A split has no examples of one class")
    return assignments


def prepare(config, output_dir):
    dataset_path = (HERE / config["dataset"]).resolve()
    rows, digest = load_rows(dataset_path)
    groups = make_groups(rows)
    assignments = stratified_group_split(rows, groups, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    with (output_dir / "splits.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "split", "gold_label"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"row_id": row["row_id"], "split": assignments[row["row_id"]], "gold_label": row["gold_label"]})
    summary = {
        "dataset": str(dataset_path), "dataset_sha256": digest, "seed": config["seed"],
        "group_columns": list(GROUP_COLUMNS) + ["subject/relation/object"],
        "group_count": len(groups),
        "splits": {name: {"total": sum(assignments[i] == name for i in assignments),
                          "class_0": sum(assignments[i] == name and rows[i]["gold_label"] == 0 for i in assignments),
                          "class_1": sum(assignments[i] == name and rows[i]["gold_label"] == 1 for i in assignments)}
                   for name in ("train", "validation", "test")},
    }
    (output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return rows, assignments


def load_saved_splits(config, output_dir):
    saved_config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
    if config != saved_config:
        raise ValueError("Configuration changed since training; use the saved run_config.json")
    rows, digest = load_rows(HERE / config["dataset"])
    summary = json.loads((output_dir / "split_summary.json").read_text(encoding="utf-8"))
    if digest != summary["dataset_sha256"]:
        raise ValueError("Dataset changed since splits were saved")
    with (output_dir / "splits.csv").open("r", encoding="utf-8", newline="") as handle:
        assignments = {int(r["row_id"]): r["split"] for r in csv.DictReader(handle)}
    if set(assignments) != set(range(len(rows))):
        raise ValueError("Saved split manifest does not match dataset rows")
    for group in make_groups(rows):
        if len({assignments[i] for i in group}) != 1:
            raise ValueError("Related triples cross saved splits")
    return rows, assignments
