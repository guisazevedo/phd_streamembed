# IMPORTS *****************************************************************************************
# External libraries
import torch
from torch import nn


# Main classes ************************************************************************************
class MeanPoolLinear(nn.Module):
    """
    Baseline: average the tile embeddings of a slide and classify the mean with one linear layer.

    * Arguments:
    in_dim: The embedding dimension of a tile
    n_classes: The number of classes
    dropout: The dropout applied to the slide vector
    """

    def __init__(self, in_dim: int, n_classes: int, dropout: float = 0.25):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x [B, N, dim], mask [B, N] (True = real tile). Returns logits [B, K] and uniform weights [B, N]."""
        weights = mask.float() / mask.float().sum(1, keepdim=True)
        slide_vector = (x * weights.unsqueeze(-1)).sum(1)

        return self.classifier(self.dropout(slide_vector)), weights


class GatedABMIL(nn.Module):
    """
    Gated attention multiple-instance learning (Ilse et al., 2018).

    A shared layer projects every tile embedding; a gated attention network (tanh branch times
    sigmoid branch) scores each tile; the scores are normalised with a softmax over the real tiles of
    the slide (padding is masked) and the attention-weighted sum is the slide vector that a linear layer
    classifies.

    * Arguments:
    in_dim: The embedding dimension of a tile
    n_classes: The number of classes
    hidden: The width of the projection
    attention_dim: The width of the attention branches
    dropout: The dropout applied after the projection and inside the attention branches
    """

    def __init__(self, in_dim: int, n_classes: int, hidden: int = 512, attention_dim: int = 256, dropout: float = 0.25):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.attention_v = nn.Sequential(nn.Linear(hidden, attention_dim), nn.Tanh(), nn.Dropout(dropout))
        self.attention_u = nn.Sequential(nn.Linear(hidden, attention_dim), nn.Sigmoid(), nn.Dropout(dropout))
        self.attention_score = nn.Linear(attention_dim, 1)
        self.classifier = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x [B, N, dim], mask [B, N] (True = real tile). Returns logits [B, K] and attention [B, N] (sums to 1 over real tiles)."""
        h = self.project(x)
        scores = self.attention_score(self.attention_v(h) * self.attention_u(h)).squeeze(-1).float()
        attention = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=1)
        slide_vector = (attention.unsqueeze(-1).to(h.dtype) * h).sum(1)

        return self.classifier(slide_vector), attention


# Main function ***********************************************************************************
def build_model(name: str, in_dim: int, n_classes: int, **kwargs) -> nn.Module:
    """
    Build a model by name: "linear" (MeanPoolLinear) or "abmil" (GatedABMIL).

    * Arguments:
    name: The model name
    in_dim: The embedding dimension of a tile
    n_classes: The number of classes (the number of classes in the label sheet)
    kwargs: Passed to the model (hidden, attention_dim, dropout)
    """
    if name == "linear":
        return MeanPoolLinear(in_dim, n_classes, **{k: v for k, v in kwargs.items() if k == "dropout"})
    if name == "abmil":
        return GatedABMIL(in_dim, n_classes, **kwargs)
    raise ValueError(f"Unknown model {name!r}; choose 'linear' or 'abmil'")
