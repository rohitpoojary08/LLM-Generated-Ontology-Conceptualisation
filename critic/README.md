# Grounding-head training and evaluation

This pipeline fine-tunes `microsoft/deberta-v3-small` as a grounding head. It predicts whether a competency question matches a subject–relation–object triple. The input dataset is `../Datasets/Grounding_Traininig_Dataset.csv`; it is never modified. The model receives the competency question and `Subject: ... Relation: ... Object: ...`. The target is `gold_label` (`0` or `1`). `source` and `theme` are used only for splitting and are not model inputs.

## Run

From `critic/`, with Python 3.10+:

```powershell
.venv\Scripts\python train.py --output-dir artifacts_grounding
.venv\Scripts\python evaluate.py --output-dir artifacts_grounding
```

## Split protocol

`config.json` retains the requested 70% train, 15% validation, and 15% test proportions. It assigns entire themes separately inside each source, keeps every record, and includes all three sources in every partition. The exact theme allocation is in `theme_splits`.

Themes are indivisible and vary in size, particularly in CQ2Term where there are only six themes. Therefore the actual row counts are the closest available theme-level allocation, rather than exact 70/15/15 in every source. `artifacts_grounding/splits.csv` records every assignment; `artifacts_grounding/split_summary.json` records exact counts, themes, labels, and the SHA-256 of the dataset. Evaluation rejects a changed CSV or a manifest that differs from the configured allocation.

The model uses AdamW with linear warmup and decay, gradient clipping, validation macro F1 for checkpoint selection, and early stopping. Test metrics are generated only after training from the best validation checkpoint.
