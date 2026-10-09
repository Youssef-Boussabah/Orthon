import orthon as ortho
from orthon._state import _PreparedLoad, _commit_prepared

from ._binding import _capture_bindings, _scaled_add_inplace


class SGD:
    """Stochastic gradient descent: param -= lr * grad."""

    def __init__(self, params, lr: float, weight_decay=0.0, decay_bias=False):
        self._bindings = _capture_bindings(params)
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
        for binding in self._bindings:
            p = binding.parameter
            if p.grad is not None:
                p.zero_grad()

    def step(self):
        with ortho.no_grad():
            for binding in self._bindings:
                p = binding.parameter
                if p.grad is not None:
                    if self._should_decay(p):
                        _scaled_add_inplace(p, p, -self._lr * self._weight_decay)
                    _scaled_add_inplace(p, p.grad, -self._lr)

    def state_dict(self):
        return {
            "lr": self._lr,
            "weight_decay": self._weight_decay,
            "decay_bias": self._decay_bias,
        }

    def load_state_dict(self, state, map_location=None):
        _commit_prepared([self._prepare_load_state_dict(state, map_location)])

    def _prepare_load_state_dict(self, state, map_location=None, placement=None):
        lr = state["lr"]
        weight_decay = state.get("weight_decay", 0.0)
        decay_bias = state.get("decay_bias", False)
        # The direct SGD loader ignores map_location: its state
        # is configuration only. File transactions supply placement facts and
        # validate the requested target without moving bound owners.
        if placement is not None and map_location is not None:
            for binding in self._bindings:
                if map_location != placement(binding.parameter):
                    raise ValueError("SGD map_location conflicts with bound parameter device")
        previous = self._lr, self._weight_decay, self._decay_bias

        def commit():
            self._lr, self._weight_decay, self._decay_bias = lr, weight_decay, decay_bias

        def rollback():
            self._lr, self._weight_decay, self._decay_bias = previous

        return _PreparedLoad(commit, rollback)

    def _rebind_parameters(self, params):
        """Private AMP bridge: validate all replacement owners before publication."""
        prepared = _capture_bindings(params)
        if len(prepared) != len(self._bindings):
            raise ValueError("SGD rebind parameter count mismatch")
        if any(old.parameter.shape != new.parameter.shape
               for old, new in zip(self._bindings, prepared)):
            raise ValueError("SGD rebind parameter shape mismatch")
        self._bindings = prepared
