import orthon as ortho


class SGD:
    """Stochastic gradient descent: param -= lr * grad."""

    def __init__(self, params, lr: float, weight_decay=0.0, decay_bias=False):
        self._params = list(params)
        self._lr = lr
        self._weight_decay = weight_decay
        self._decay_bias = decay_bias

    def get_lr(self):
        return self._lr

    def set_lr(self, lr):
        self._lr = lr

    def _should_decay(self, p):
        return (
            self._weight_decay != 0.0
            and p.requires_grad
            and (self._decay_bias or len(p.shape) > 1)
        )

    def zero_grad(self):
        for p in self._params:
            if p.grad is not None:
                p.zero_grad()

    def step(self):
        with ortho.no_grad():
            for p in self._params:
                if p.grad is not None:
                    if self._should_decay(p):
                        p.add_(p, alpha=-self._lr * self._weight_decay)
                    p.add_(p.grad, alpha=-self._lr)

    def state_dict(self):
        return {
            "lr": self._lr,
            "weight_decay": self._weight_decay,
            "decay_bias": self._decay_bias,
        }

    def load_state_dict(self, state, map_location=None):
        self._lr = state["lr"]
        self._weight_decay = state.get("weight_decay", 0.0)
        self._decay_bias = state.get("decay_bias", False)
