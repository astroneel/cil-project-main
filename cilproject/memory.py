"""Module contains functionality
For passing memory between phases."""
import sklearn.cluster
import torch


def _split_by_label(dataset, label):
    return [x for x, y in dataset if y == label]


def add_memory(
    memory, dataset, embedder, label_offset: int, max_per_class=5, device="cpu", **_
):
    for label in range(10):
        chosen = _split_by_label(dataset, label)[:max_per_class]
        memory[label + label_offset] += chosen
    return memory


def add_kmeans_memory(
    memory,
    dataset,
    embedder,
    label_offset: int,
    max_per_class=5,
    device="cpu",
    labels=range(10),
    **_,
):
    for label in labels:
        splits = _split_by_label(dataset, label)
        splits = torch.stack(splits).numpy()
        kmeans = sklearn.cluster.KMeans(
            n_clusters=max_per_class, random_state=0, n_init="auto"
        ).fit(splits)
        # find indices of centers and add only those to memory
        centers = list(kmeans.cluster_centers_)
        centers = [torch.from_numpy(x).to(device) for x in centers]
        memory[label + label_offset] += centers
    return memory


def get_memory_function(memory_type: str):
    if memory_type == "kmeans":
        return add_kmeans_memory
    else:
        return add_memory
