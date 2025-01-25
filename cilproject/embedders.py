"""Stores code for instantiating embedders."""
from torch import nn
import torch
import timm


class FrozenEmbedder(nn.Module):
    """An embedder that does not update its weights during training."""

    def __init__(self, embedder):
        super().__init__()
        self.embedder = embedder

    def __call__(self, *args, **kwargs):
        return self.embedder(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Forward pass through the embedder."""
        return self.embedder(*args, **kwargs)

    def train(self, mode=True):
        pass


class TimmEmbedder(nn.Module):
    """An embedder that uses a pretrained timm model."""

    def __init__(self, model_name, pretrained=True, use_existing_head=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.eval()
        self.use_existing_head = use_existing_head

    def forward(self, x: torch.Tensor):
        """Forward pass through the embedder.

        N.B. We usually use use_existing_head=True for our experiments."""
        if self.use_existing_head:
            return self.model(x)
        out = self.model.forward_features(x)
        out = torch.nn.AdaptiveAvgPool2d((1, 1))(out)
        out = out.view(out.size(0), -1)
        return out

    def train(self, mode=True):
        self.model.train(mode)


class EmbedderCache(nn.Module):
    """A cache for the embedder."""

    def __init__(self, embedder):
        super().__init__()
        self.embedder = embedder
        self.cache = {}

    def forward(self, x: torch.Tensor):
        """Forward pass through the embedder."""
        return self.embedder(x)

    def train(self, mode=True):
        self.embedder.train(mode)


def get_embedder(device: str, model_name: str, **kwargs):
    """Returns the embedder."""
    return TimmEmbedder(model_name, **kwargs).to(device)


if __name__ == "__main__":
    embedder = get_embedder(
        "cpu", "repvit_m3.dist_in1k", pretrained=True, use_existing_head=False
    )
    random_input = torch.randn(1, 3, 34, 34)
    out = embedder(random_input)
    out = torch.nn.AdaptiveAvgPool2d((1, 1))(out)
    out = out.view(out.size(0), -1)
    out = embedder.model.head(out)
    embedder_2 = get_embedder(
        "cpu", "repvit_m3.dist_in1k", pretrained=True, use_existing_head=True
    )
    out_2 = embedder_2(random_input)
    # calc difference
    diff = torch.sum(out - out_2)
    print(diff)

    # out = embedder.model.head(out)
    # check memory size of the model
    # torch.save(embedder.state_dict(), "embedder.pt")
