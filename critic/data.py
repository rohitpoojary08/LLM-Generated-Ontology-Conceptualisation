"""Load the grounding dataset and assign source-stratified theme splits."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPLITS = ("train", "validation", "test")
REQUIRED = (
    "original_triple", "subject", "relation", "object", "competency_question",
    "gold_label", "theme", "source",
)


def load_rows(dataset_path):
    path = Path(dataset_path).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(REQUIRED) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Dataset is empty")
    for row_id, row in enumerate(rows):
        if any(not row[column].strip() for column in REQUIRED):
            raise ValueError(f"Empty required field on CSV row {row_id + 2}")
        if row["gold_label"] not in {"0", "1"}:
            raise ValueError(f"Invalid gold_label on CSV row {row_id + 2}")
        row["row_id"] = row_id
        row["gold_label"] = int(row["gold_label"])
    return rows, digest


def split_rows(rows, config):
    """Assign complete source/theme strata to a split.

    Themes are selected in config.  Every row with a selected source and theme
    goes to the same partition, so validation and test contain unseen themes.
    """
    configured = config["theme_splits"]
    sources = set(row["source"] for row in rows)
    if set(configured) != sources:
        raise ValueError("theme_splits must contain exactly the sources in the dataset")

    assignment = {}
    for source, themes in configured.items():
        if set(themes) != set(SPLITS):
            raise ValueError(f"theme_splits[{source}] must contain train, validation, and test")
        lookup = {}
        for split, values in themes.items():
            for theme in values:
                if theme in lookup:
                    raise ValueError(f"Theme {theme!r} appears more than once for {source}")
                lookup[theme] = split
        present = {row["theme"] for row in rows if row["source"] == source}
        if set(lookup) != present:
            missing, extra = present - set(lookup), set(lookup) - present
            raise ValueError(f"Theme configuration for {source} differs from CSV; missing={sorted(missing)}, extra={sorted(extra)}")
        for row in rows:
            if row["source"] == source:
                assignment[row["row_id"]] = lookup[row["theme"]]
    return assignment


def summary_for(rows, assignment, config, digest, dataset_path):
    sources = config["theme_splits"]
    return {
        "dataset": str(dataset_path),
        "dataset_sha256": digest,
        "seed": config["seed"],
        "strategy": "Each source is split by whole themes. Train, validation, and test use disjoint themes within a source.",
        "fractions_requested": {name: config[f"{name}_fraction"] for name in SPLITS},
        "splits": {
            name: {"total": sum(assignment[row["row_id"]] == name for row in rows),
                   "class_0": sum(assignment[row["row_id"]] == name and row["gold_label"] == 0 for row in rows),
                   "class_1": sum(assignment[row["row_id"]] == name and row["gold_label"] == 1 for row in rows)}
            for name in SPLITS
        },
        "sources": {
            source: {
                "total": sum(row["source"] == source for row in rows),
                "splits": {
                    name: {
                        "total": sum(row["source"] == source and assignment[row["row_id"]] == name for row in rows),
                        "class_0": sum(row["source"] == source and assignment[row["row_id"]] == name and row["gold_label"] == 0 for row in rows),
                        "class_1": sum(row["source"] == source and assignment[row["row_id"]] == name and row["gold_label"] == 1 for row in rows),
                        "themes": dict(sorted(Counter(row["theme"] for row in rows if row["source"] == source and assignment[row["row_id"]] == name).items())),
                    } for name in SPLITS
                },
            } for source in sources
        },
    }


def prepare(config, output_dir):
    dataset_path = (HERE / config["dataset"]).resolve()
    rows, digest = load_rows(dataset_path)
    assignment = split_rows(rows, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    with (output_dir / "splits.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("row_id", "split", "gold_label", "source", "theme"))
        writer.writeheader()
        for row in rows:
            writer.writerow({"row_id": row["row_id"], "split": assignment[row["row_id"]],
                             "gold_label": row["gold_label"], "source": row["source"], "theme": row["theme"]})
    summary = summary_for(rows, assignment, config, digest, dataset_path)
    (output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return rows, assignment


def load_saved_splits(config, output_dir):
    saved_config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
    if config != saved_config:
        raise ValueError("Configuration changed since training; use the saved run_config.json")
    rows, digest = load_rows(HERE / config["dataset"])
    summary = json.loads((output_dir / "split_summary.json").read_text(encoding="utf-8"))
    if digest != summary["dataset_sha256"]:
        raise ValueError("Dataset changed since splits were saved")
    with (output_dir / "splits.csv").open("r", encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    if len(manifest) != len(rows) or {int(item["row_id"]) for item in manifest} != set(range(len(rows))):
        raise ValueError("Saved split manifest does not match dataset rows")
    assignment = {int(item["row_id"]): item["split"] for item in manifest}
    expected = split_rows(rows, config)
    if assignment != expected:
        raise ValueError("Saved split manifest differs from the configured theme split")
    return rows, assignment
