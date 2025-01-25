"""Script used for running training experiments."""
import collections

import cilproject.classifiers as classifiers
import cilproject.dataset as dataset
import cilproject.embedders as embedders
import cilproject.memory as memory
import cilproject.predictions as predictions
import cilproject.evaluate as evaluate
import cilproject.models as models
import cilproject.train as train
import torch
import torch.utils.data
import typer


def per_class_random_split(dataset, split_proportion):
    """Splits the dataset into train and val such that each class is split
    proportionally."""
    train_dataset = []
    val_dataset = []
    for label in range(10):
        label_dataset = torch.utils.data.Subset(
            dataset, [i for i, (_, y) in enumerate(dataset) if y == label]
        )
        train_size = int(split_proportion * len(label_dataset))
        train_dataset.extend(
            [
                items
                for items in torch.utils.data.Subset(label_dataset, range(train_size))
            ]
        )
        val_dataset.extend(
            [
                items
                for items in torch.utils.data.Subset(
                    label_dataset, range(train_size, len(label_dataset))
                )
            ]
        )
    return train_dataset, val_dataset


class LabeledLeaderboardDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, preds, mask):
        self.dataset = dataset
        self.preds = preds
        self.mask = mask

    def __getitem__(self, mask_idx):
        idx = [i for i, x in enumerate(self.mask) if x][mask_idx]
        x, _ = self.dataset[idx]
        y = self.preds[idx]
        return x, y

    def __len__(self):
        return len([x for x in self.mask if x])


def load_val_dataset(og_data_folder_path: str, phase: int, preds_path: str):
    """Loads predictions from val leaderboard and uses these
    to gain additional training data"""
    val_dataset = dataset.LeaderboardValDataset(
        f"{og_data_folder_path}/Val",
        dataset.get_imagenet_transform(),
    )
    with open(f"{preds_path}/result_10.txt", "r", encoding="utf-8") as f:
        preds = f.readlines()
    preds = [int(x.split(" ")[1]) for x in preds]
    mask = [x in range((phase - 1) * 10, phase * 10) for x in preds]

    return LabeledLeaderboardDataset(val_dataset, preds, mask)


def run_experiment(
    data_folder_path: str = typer.Argument(...),
    device: str = typer.Option("mps"),
    random_seed: int = typer.Option(42),
    embedding_model: str = typer.Option("hf_hub:timm/tiny_vit_21m_224.in1k"),
    memory_type: str = typer.Option("kmeans"),
    train_phase_with_memory: bool = typer.Option(False),
    pred_path: str = typer.Option(None),
    phase_model: str = typer.Option(None),
    common_model: str = typer.Option(None),
    aggregation: str = typer.Option(None),
    save_preds: bool = typer.Option(False),
    train_type: str = typer.Option("contrastive"),
    disable_val: bool = typer.Option(False),
    num_epochs: int = typer.Option(10),
    classifier_name: str = typer.Option("linear"),
):
    """Runs an experiment."""
    torch.manual_seed(random_seed)
    if aggregation is None:
        aggregation = "concat"
    embedder = embedders.get_embedder(
        device, embedding_model, use_existing_head=common_model != "timm"
    )
    embedding_size = 1000
    if common_model == "timm":
        common_model = embedder.model.head
        embedding_size = 512
    embedder = embedders.EmbedderCache(embedder)
    history = collections.defaultdict(list)
    val_history = collections.defaultdict(list)
    model = models.get_model(
        phase_model,
        aggregation,
        common_model,
        embedding_size=embedding_size,
        output_size=1000,
    )
    print(model)
    model.to(device)
    scores = []
    for phase in range(1, 11):
        train_dataset = dataset.get_train_dataset(
            phase, dataset.get_imagenet_transform(), data_dir=data_folder_path
        )
        if not disable_val:
            # train_ds, val_ds = torch.utils.data.random_split(
            #     train_dataset,
            #     [
            #         int(0.8 * len(train_dataset)),
            #         len(train_dataset) - int(0.8 * len(train_dataset)),
            #     ],
            # )
            train_ds, val_ds = per_class_random_split(train_dataset, 0.8)
        else:
            train_ds = train_dataset
        # additional_data = load_val_dataset(f"{data_folder_path}", phase, pred_path)
        # train_ds = torch.utils.data.ConcatDataset([train_ds, additional_data])
        train_ds = dataset.EmbeddedDataset(
            train_ds,
            embedder,
            device=device,
            save_path=f"{data_folder_path}/train_{phase}.pt",
        )
        if not disable_val:
            val_ds = dataset.EmbeddedDataset(
                val_ds,
                embedder,
                device=device,
                save_path=f"{data_folder_path}/val_{phase}.pt",
            )
        if model.per_phase_models is not None and train_type == "contrastive":
            if train_phase_with_memory:
                model = train.train_model_contrastive(
                    model,
                    train_ds,
                    phase=phase,
                    history=history,
                    epochs=num_epochs,
                    device=device,
                    lr=1e-4,
                )
            else:
                model = train.train_model_contrastive(
                    model,
                    train_ds,
                    phase=phase,
                    epochs=num_epochs,
                    device=device,
                    lr=1e-4,
                )
        elif model.per_phase_models is not None and train_type == "supervised":
            model = train.train_model_cross_entropy(
                model,
                train_ds,
                phase=phase,
                history=history,
                epochs=10,
                device=device,
                lr=1e-4,
            )
        if not disable_val:
            val_history = memory.add_memory(
                val_history,
                val_ds,
                embedder=embedder,
                label_offset=(phase - 1) * 10,
                max_per_class=1000,
                device=device,
            )
        memory_func = memory.get_memory_function(memory_type)
        history = memory_func(
            history,
            train_ds,
            label_offset=(phase - 1) * 10,
            embedder=embedder,
            max_per_class=5,
            device=device,
        )
        if aggregation == "calibrated_add":
            model = train.calibrate_model(
                model,
                phase=phase,
                history=history,
                epochs=2,
                device=device,
                lr=1e-3,
            )
        model_history = models.get_model_embedded_history(model, history, device=device)
        classifier = classifiers.train_classifier(
            classifier_name=classifier_name,
            history=model_history,
        )
        if not disable_val:
            model_val_history = models.get_model_embedded_history(
                model, val_history, device=device
            )
            score = evaluate.evaluate_classifier(
                model_val_history,
                classifier,
            )
            scores.append(score)
            print(f"Phase {phase} score: {score}")
        if save_preds:
            print(f"Saving predictions for phase {phase} at {pred_path}")
            predictions.save_predictions(
                dataset.LeaderboardValDataset(
                    f"{data_folder_path}/Test",
                    dataset.get_imagenet_transform(),
                ),
                embedder=embedder,
                model=model,
                classifier=classifier,
                pred_path=pred_path,
                device="mps",
                phase=phase,
            )
        print(f"Phase {phase}: {len(train_dataset)} images")

    if not disable_val:
        print(f"Average score: {sum(scores) / len(scores)}")


if __name__ == "__main__":
    typer.run(run_experiment)
