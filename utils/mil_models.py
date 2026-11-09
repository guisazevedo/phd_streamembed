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


class ClamMB(nn.Module):
    """
    Clustering-constrained attention multiple-instance learning, multi-branch (CLAM-MB; Lu et al., 2021).

    A shared layer projects every tile embedding. A gated attention network gives every tile one score per
    class; for each class the scores are normalised with a softmax over the real tiles of the slide (padding
    is masked), so that every class has its own attention-weighted slide vector. A separate linear layer per
    class turns its slide vector into the logit of that class. During training an instance-level clustering
    loss is added: for the true class, the tiles with the highest attention of that class are given the
    pseudo-label 1 and those with the lowest attention the pseudo-label 0; for every other class, the tiles
    with the highest attention of that class are given the pseudo-label 0. One small classifier per class is
    trained on these pseudo-labels, which pushes the projection to separate the tiles that the attention
    selects. The pseudo-labels come from the model itself and are not tile ground truth.

    * Arguments:
    in_dim: The embedding dimension of a tile
    n_classes: The number of classes
    hidden: The width of the projection
    attention_dim: The width of the attention branches
    dropout: The dropout applied after the projection and inside the attention branches
    k_sample: The number of highest- (and, for the true class, lowest-) attention tiles used for the instance loss
    bag_weight: The weight of the slide-level loss in the total loss; the instance loss gets 1 - bag_weight
    """

    # train_model passes the labels of the batch to forward() and adds the instance loss to the loss
    needs_labels = True

    def __init__(self, in_dim: int, n_classes: int, hidden: int = 512, attention_dim: int = 256, dropout: float = 0.25,
                 k_sample: int = 8, bag_weight: float = 0.7):
        super().__init__()
        self.n_classes, self.k_sample, self.bag_weight = n_classes, k_sample, bag_weight
        self.project = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.attention_v = nn.Sequential(nn.Linear(hidden, attention_dim), nn.Tanh(), nn.Dropout(dropout))
        self.attention_u = nn.Sequential(nn.Linear(hidden, attention_dim), nn.Sigmoid(), nn.Dropout(dropout))
        self.attention_score = nn.Linear(attention_dim, n_classes)
        self.classifiers = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(n_classes)])
        self.instance_classifiers = nn.ModuleList([nn.Linear(hidden, 2) for _ in range(n_classes)])
        self.instance_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x [B, N, dim], mask [B, N] (True = real tile), labels [B] (optional, training only).

        Returns logits [B, K] and the attention [B, N] of the predicted class (sums to 1 over the real tiles).
        With labels, the instance loss is stored in self.instance_loss; without, it is None.
        """

        # STEP 1: Class-wise attention over the real tiles
        h = self.project(x)
        scores = self.attention_score(self.attention_v(h) * self.attention_u(h)).float()          # [B, N, K]
        attention = torch.softmax(scores.masked_fill(~mask.unsqueeze(-1), float("-inf")), dim=1)

        # STEP 2: One slide vector and one logit per class
        slide_vectors = torch.einsum("bnk,bnh->bkh", attention.to(h.dtype), h)                      # [B, K, hidden]
        logits = torch.cat([self.classifiers[k](slide_vectors[:, k]) for k in range(self.n_classes)], dim=1)

        # STEP 3: The instance loss (training) and the attention of the predicted class (for inspection)
        self.instance_loss = self._instance_loss(h, attention, mask, labels) if labels is not None else None
        predicted = logits.detach().argmax(1)

        return logits, attention[torch.arange(len(x), device=x.device), :, predicted]

    def _instance_loss(self, h: torch.Tensor, attention: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Mean cross-entropy of the instance classifiers on the attention-selected tiles (see the class docstring)."""

        losses = []
        for i in range(len(h)):
            # STEP 1: The real tiles of this slide; a bag of one tile cannot be split into top and bottom tiles
            real = mask[i].nonzero().squeeze(-1)
            if len(real) < 2:
                continue
            k = min(self.k_sample, len(real) // 2)
            tiles = h[i, real]

            for c in range(self.n_classes):
                # STEP 2: Pseudo-labelled tiles of this class's attention branch
                weights = attention[i, real, c]
                top = weights.topk(k).indices
                if c == int(labels[i]):
                    index = torch.cat([top, (-weights).topk(k).indices])
                    target = torch.cat([torch.ones(k), torch.zeros(k)]).long().to(h.device)
                else:
                    index = top
                    target = torch.zeros(k, dtype=torch.long, device=h.device)
                losses.append(nn.functional.cross_entropy(self.instance_classifiers[c](tiles[index]).float(), target))

        return torch.stack(losses).mean() if losses else h.new_zeros((), dtype=torch.float32)


# Main function ***********************************************************************************
def build_model(name: str, in_dim: int, n_classes: int, **kwargs) -> nn.Module:
    """
    Build a model by name: "linear" (MeanPoolLinear), "abmil" (GatedABMIL) or "clam_mb" (ClamMB).

    * Arguments:
    name: The model name
    in_dim: The embedding dimension of a tile
    n_classes: The number of classes (the number of classes in the label sheet)
    kwargs: Passed to the model (hidden, attention_dim, dropout; for "clam_mb" also k_sample and bag_weight)
    """
    if name == "linear":
        return MeanPoolLinear(in_dim, n_classes, **{k: v for k, v in kwargs.items() if k == "dropout"})
    if name == "abmil":
        return GatedABMIL(in_dim, n_classes, **kwargs)
    if name == "clam_mb":
        return ClamMB(in_dim, n_classes, **kwargs)
    raise ValueError(f"Unknown model {name!r}; choose 'linear', 'abmil' or 'clam_mb'")
