"""Stores methods for evaluation."""
import numpy as np
import torch


def evaluate_classifier(val_data: dict, classifier):
    X = np.concatenate(list(val_data.values()))
    y = np.concatenate([[k] * len(v) for k, v in val_data.items()])
    return classifier.score(X, y)
