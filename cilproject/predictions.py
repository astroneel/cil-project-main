"""Contains methods for making/saving predictions with the models."""
import numpy as np
import torch

BATCH_SIZE = 64


def save_predictions(
    dataset, embedder, model, classifier, pred_path, device="cpu", phase=1
):
    """Saves predictions for the given dataset."""
    lines = []
    xs = [x for x, _ in dataset]
    batched_xs = []
    embeddings = []
    for i, x in enumerate(xs):
        batched_xs.append(x)
        if len(batched_xs) == BATCH_SIZE or i == len(xs) - 1:
            with torch.no_grad():
                embedding = embedder(torch.stack(batched_xs).to(device)).squeeze()
                embedding = model(embedding).cpu()
                embeddings.extend(embedding)
            batched_xs = []
    for j, (_, label) in enumerate(dataset):
        embedding = embeddings[j].numpy()
        pred_classes = classifier.predict_proba(embedding.reshape(1, -1))[0]
        lines.append(f"{label.split('/')[-1]} {str(np.argmax(pred_classes))}")
    with open(f"{pred_path}/result_{phase}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved predictions for phase {phase}.")
