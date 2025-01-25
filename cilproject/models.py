"""Defines any models used in the project."""
import torch
import torch.utils.data
from torch import nn
import tqdm
import abc
from typing import Sequence
from collections import defaultdict


class PerPhaseModel(nn.Module, abc.ABC):
    def __init__(
        self,
    ):
        super().__init__()

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass


class LinearPerPhaseModel(PerPhaseModel):
    def __init__(self, num_classes: int, embedding_size: int):
        super().__init__()
        self.linear = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        return self.linear(x)


class NNPerPhaseModel(PerPhaseModel):
    def __init__(
        self, num_classes: int, embedding_size: int, hidden_size: int, num_layers: int
    ):
        super().__init__()
        self.linear = nn.Linear(embedding_size, hidden_size)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        )
        self.final = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.linear(x)
        x = torch.relu(x)
        for layer in self.layers:
            x = layer(x)
            x = torch.relu(x)
        return self.final(x)


class NormMlpClassifierHead(PerPhaseModel):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.norm_layer = nn.LayerNorm(in_features)
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.norm_layer(x)
        x = torch.tanh(x)
        return self.linear(x)


class NormMLPResidualHead(PerPhaseModel):
    def __init__(self, base_model, embedding_size: int, output_size: int):
        super().__init__()
        self.base_model = base_model
        self.residual_model = NormMlpClassifierHead(
            in_features=embedding_size,
            num_classes=output_size,
        )

    def forward(self, x):
        with torch.no_grad():
            x_base = self.base_model(x)
        return x_base + self.residual_model(x)


class NormMLPPhasedResidualHead(PerPhaseModel):
    def __init__(self, base_model, embedding_size: int, phase: int):
        super().__init__()
        self.base_model = base_model
        self.residual_model = NormMlpClassifierHead(
            in_features=embedding_size,
            num_classes=embedding_size // 10,
        )
        self.idx_mod = embedding_size // 10
        self.phase = phase

    def forward(self, x):
        with torch.no_grad():
            x_base = self.base_model(x)
        partial_residual = self.residual_model(x)
        zeros = torch.zeros_like(x_base)
        zeros[
            :, self.phase * self.idx_mod : (self.phase + 1) * self.idx_mod
        ] = partial_residual
        return x_base + zeros
        # return x_base


class Aggregator(nn.Module, abc.ABC):
    def __init__(
        self,
    ):
        super().__init__()

    @abc.abstractmethod
    def forward(self, common_embedding, per_phase_embeddings):
        pass


class ConcatAggregator(Aggregator):
    def forward(self, common_embedding, per_phase_embeddings):
        if common_embedding is None:
            return torch.cat(per_phase_embeddings, dim=1)
        elif per_phase_embeddings is None:
            return common_embedding
        return torch.cat([common_embedding] + per_phase_embeddings, dim=1)


class AddAggregator(Aggregator):
    def __init__(self, normalize: bool = False):
        super().__init__()
        self.normalize = normalize

    def forward(self, common_embedding, per_phase_embeddings):
        if self.normalize and per_phase_embeddings:
            per_phase_embeddings = [
                nn.functional.normalize(x, dim=1) for x in per_phase_embeddings
            ]
        if self.normalize and common_embedding:
            common_embedding = nn.functional.normalize(common_embedding, dim=1)
        if common_embedding is None:
            return sum(per_phase_embeddings)
        elif per_phase_embeddings is None or len(per_phase_embeddings) == 0:
            return common_embedding
        per_phase_embeddings = [
            per_phase_embedding - common_embedding
            for per_phase_embedding in per_phase_embeddings
        ]
        # We can try different ways of combining...
        # out = common_embedding + sum(per_phase_embeddings)
        # out = nn.functional.normalize(out, dim=1)
        out = common_embedding + sum(per_phase_embeddings) / len(per_phase_embeddings)
        return out


class AddConcatAggregator(Aggregator):
    def __init__(self):
        super().__init__()

    def forward(self, common_embedding, per_phase_embeddings):
        if common_embedding is None:
            return torch.cat(per_phase_embeddings, dim=1)
        elif per_phase_embeddings is None:
            return common_embedding
        per_phase_embeddings = [
            per_phase_embedding - common_embedding
            for per_phase_embedding in per_phase_embeddings
        ]
        mod_embedding = sum([common_embedding] + per_phase_embeddings) / (
            len(per_phase_embeddings) + 1
        )
        out = torch.cat([common_embedding, mod_embedding], dim=1)
        return out


class CalibratedAggregator(Aggregator):
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.scale_params = nn.Parameter(torch.ones(n_classes + 1))

    def reset(self):
        nn.init.ones_(self.scale_params)

    def forward(self, common_embedding, per_phase_embeddings):
        if common_embedding is None:
            return sum(per_phase_embeddings)
        elif per_phase_embeddings is None:
            return common_embedding
        per_phase_embeddings = [
            per_phase_embedding - common_embedding
            for per_phase_embedding in per_phase_embeddings
        ]
        out = (
            torch.stack([common_embedding] + per_phase_embeddings, -1)
            @ self.scale_params[: len(per_phase_embeddings) + 1]
        )
        return out


class CommonModel(nn.Module, abc.ABC):
    def __init__(
        self,
    ):
        super().__init__()

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> torch.TensorType:
        pass


class IdModel(CommonModel):
    def forward(self, x):
        return x


class CombinedModel(nn.Module):
    """A model used to combine together different subcomponents,

    handling the logic of passing the data between them, and
    whether or not they exist."""

    def __init__(
        self,
        per_phase_models: Sequence[PerPhaseModel] | None,
        aggregator: Aggregator,
        common_model: CommonModel | None = None,
    ):
        super().__init__()
        if per_phase_models is not None:
            self.per_phase_models = nn.ModuleList(per_phase_models)
        else:
            self.per_phase_models = None
        self.aggregator: Aggregator = aggregator
        self.common_model = common_model
        self.curr_phase = 0

    def forward(self, *args, **kwargs):
        if self.common_model is None:
            common_embedding = None
        else:
            common_embedding = self.common_model(*args, **kwargs)
        if self.per_phase_models is None:
            per_phase_embeddings = None
        else:
            per_phase_embeddings = [
                per_phase_model(*args, **kwargs)
                for per_phase_model in self.per_phase_models
            ]
        return self.aggregator(common_embedding, per_phase_embeddings)


class AdaptiveCombinedModel(CombinedModel):
    """Model similar to combined model, but which only updates the first
    phase model during training to learn domain-specific features."""

    def __init__(
        self,
        per_phase_models: Sequence[PerPhaseModel] | None,
        aggregator: Aggregator,
        common_model: CommonModel | None = None,
    ):
        super().__init__(per_phase_models, aggregator, common_model)
        self.per_phase_models = nn.ModuleList(per_phase_models)
        self.curr_phase = 0

    def forward(self, *args, **kwargs):
        if self.common_model is None:
            common_embedding = None
        else:
            common_embedding = self.common_model(*args, **kwargs)
        if self.per_phase_models is None:
            per_phase_embeddings = None
        else:
            per_phase_embeddings = [
                per_phase_model(*args, **kwargs)
                for i, per_phase_model in enumerate(self.per_phase_models)
                if i == 0
            ]
        out = self.aggregator(common_embedding, per_phase_embeddings)
        ## normalize the output
        # out = nn.functional.normalize(out, dim=1)
        return out


def get_model(
    per_phase_model: str | None,
    aggregator: str,
    common_model: str | torch.nn.Module | None = None,
    embedding_size: int = 1000,
    output_size: int = 1000,
):
    """Returns the model."""
    if per_phase_model == "adaptive":
        return AdaptiveCombinedModel(
            per_phase_models=[
                NormMLPResidualHead(
                    base_model=IdModel(),
                    embedding_size=embedding_size,
                    output_size=output_size,
                )
                for _ in range(10)
            ],
            aggregator=AddAggregator(),
            common_model=IdModel(),
        )
    if common_model == "id":
        common = IdModel()
    elif common_model is not str:
        common = common_model
    elif common_model is None:
        common = None
    else:
        raise ValueError(f"Unknown common model {common_model}.")
    if per_phase_model == "linear":
        phase_models = [
            LinearPerPhaseModel(num_classes=10, embedding_size=embedding_size)
            for _ in range(10)
        ]
    elif per_phase_model == "nn":
        phase_models = [
            NNPerPhaseModel(
                num_classes=10,
                embedding_size=embedding_size,
                hidden_size=100,
                num_layers=1,
            )
            for _ in range(10)
        ]
    elif per_phase_model == "timm_classifier":
        phase_models = [
            NormMlpClassifierHead(
                in_features=embedding_size,
                num_classes=10,
            )
            for _ in range(10)
        ]
    elif per_phase_model == "timm_classifier_residual":
        phase_models = [
            NormMLPResidualHead(
                base_model=common,
                embedding_size=embedding_size,
                output_size=output_size,
            )
            for _ in range(10)
        ]
    elif per_phase_model == "timm_classifier_residual_2":
        phase_models = [
            NormMLPPhasedResidualHead(
                base_model=common,
                embedding_size=embedding_size,
                phase=i,
            )
            for i in range(10)
        ]
    elif per_phase_model == "timm_classifier_pseudo_residual":
        phase_models = [
            NormMLPResidualHead(
                base_model=lambda x: torch.zeros((x.shape[0], output_size)).to(
                    x.device
                ),
                embedding_size=embedding_size,
                output_size=output_size,
            )
            for _ in range(10)
        ]
    elif per_phase_model is None:
        phase_models = None
    else:
        raise ValueError(f"Unknown per phase model {per_phase_model}.")
    if aggregator == "concat":
        agg = ConcatAggregator()
    elif aggregator == "add":
        agg = AddAggregator()
    elif aggregator == "calibrated_add":
        agg = CalibratedAggregator()
    elif aggregator == "cat_add":
        agg = AddConcatAggregator()
    else:
        raise ValueError(f"Unknown aggregator {aggregator}.")
    return CombinedModel(phase_models, agg, common)


def get_model_embedded_history(
    model: CombinedModel,
    history,
    device: str = "cpu",
):
    """Returns the history with the model's embeddings added."""
    model.eval()
    new_dict = defaultdict(list)
    for label, embeddings in history.items():
        if len(embeddings) == 0:
            print(f"Skipping {label}")
        with torch.no_grad():
            new_dict[label] = (
                model(
                    torch.stack(embeddings).to(device),
                )
                .cpu()
                .numpy()
            )
    return new_dict
