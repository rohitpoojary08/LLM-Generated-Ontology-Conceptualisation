"""Shared model inputs, reproducibility, and inference."""

import random

import numpy as np
import torch
from torch.utils.data import Dataset


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def model_inputs(row):
    question = row["competency_question"]
    triple = f'Subject: {row["subject"]} Relation: {row["relation"]} Object: {row["object"]}'
    return question, triple


class CriticDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.rows = rows
        questions, triples = zip(*(model_inputs(row) for row in rows))
        self.encodings = tokenizer(
            list(questions), list(triples), padding=False, truncation=True,
            max_length=max_length,
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        item = {key: values[index] for key, values in self.encodings.items()}
        item["labels"] = self.rows[index]["gold_label"]
        return item


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    labels, predictions, probabilities = [], [], []
    for batch in loader:
        gold = batch.pop("labels")
        inputs = {key: value.to(device) for key, value in batch.items()}
        output = model(**inputs)
        probs = torch.softmax(output.logits.float(), dim=-1)[:, 1].cpu().tolist()
        labels.extend(gold.tolist())
        probabilities.extend(probs)
        predictions.extend(int(p >= 0.5) for p in probs)
    return labels, predictions, probabilities
