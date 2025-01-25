"""Utilites for loading the dataset in all it's forms"""
import torchvision.datasets
import torchvision.transforms
import torch.utils.data as data
import torch


def get_train_dataset(phase, transform, data_dir="data"):
    """Returns the train dataset for a given phase."""
    return torchvision.datasets.ImageFolder(
        f"{data_dir}/Train/phase_{phase}", transform=transform
    )


class LeaderboardValDataset(data.Dataset):
    """The leaderboard val dataset."""

    def __init__(self, path, transform):
        self.dataset = torchvision.datasets.ImageFolder(path, transform=transform)

    def __getitem__(self, index):
        return self.dataset[index][0], self.dataset.imgs[index][0]

    def __len__(self):
        return len(self.dataset)


class LeaderboardTestDataset(data.Dataset):
    """The leaderboard test dataset."""

    def __init__(self, path, transform):
        self.dataset = torchvision.datasets.ImageFolder(path, transform=transform)

    def __getitem__(self, index):
        return self.dataset[index][0], self.dataset.imgs[index][0]

    def __len__(self):
        return len(self.dataset)


class EmbeddedDataset(data.Dataset):
    """A dataset that returns the embedding of the image."""

    def __init__(self, dataset, embedder, device="cpu", save_path=None):
        self.dataset = dataset
        self.embedder = embedder
        self.device = device
        self.save_path = save_path
        if self.save_path is not None:
            self.save_and_embed()

    def save_and_embed(self):
        """Saves the embeddings of the dataset."""
        if self.save_path is None:
            return
        embeddings = []
        dl = data.DataLoader(
            self.dataset,
            batch_size=64,
            shuffle=False,
        )
        for x, _ in dl:
            with torch.no_grad():
                embeddings.extend(self.embedder(x.to(self.device)).cpu())
        torch.save(torch.stack(embeddings), self.save_path)

    def __getitem__(self, index):
        if self.save_path is None:
            return (
                self.embedder(self.dataset[index][0].to(self.device)).cpu(),
                self.dataset[index][1],
            )
        else:
            return torch.load(self.save_path)[index], self.dataset[index][1]

    def __len__(self):
        return len(self.dataset)


def get_imagenet_transform():
    """Returns the imagenet transform."""
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.CenterCrop((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
