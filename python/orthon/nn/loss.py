import orthon as ortho
from .module import Module


class NLLLoss(Module):
    """Negative log likelihood loss for [N, C] log-probabilities."""

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, log_probs, targets):
        return ortho.nll_loss(log_probs, targets, reduction=self.reduction)
