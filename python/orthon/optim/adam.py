import os

import orthon as ortho


def _flatten_nested(values):
    if isinstance(values, (int, float)):
        return [float(values)]
    if isinstance(values, (list, tuple)):
        flat = []
        for value in values:
            flat.extend(_flatten_nested(value))
        return flat
    return [float(values)]


def _tensor_stats(tensor):
    values = _flatten_nested(tensor.tolist())
    if not values:
        return {"max": 0.0, "mean": 0.0}
    abs_values = [abs(value) for value in values]
    return {
        "max": max(abs_values),
        "mean": sum(abs_values) / len(abs_values),
    }


def _debug_stats(index, shape, grad, m, v, v_hat, sqrt_v_hat, denom, step):
    print(f"adam debug param {index} shape={list(shape)}")
    for name, tensor in [
        ("grad", grad),
        ("m", m),
        ("v", v),
        ("v_hat", v_hat),
        ("sqrt_v_hat", sqrt_v_hat),
        ("denom", denom),
        ("update", step),
    ]:
        stats = _tensor_stats(tensor)
        print(f"  {name}: max={stats['max']:.6e} mean={stats['mean']:.6e}")


class Adam:
    """Adam optimizer: adaptive moment estimation.

    m = beta1*m + (1-beta1)*grad
    v = beta2*v + (1-beta2)*grad^2
    p -= lr * (m/bc1) / (sqrt(v/bc2) + eps)
    """

    def __init__(
        self,
        params,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        decay_bias=False,
    ):
        self._params = list(params)
        self._lr     = lr
        self._beta1, self._beta2 = betas
        self._eps    = eps
        self._weight_decay = weight_decay
        self._decay_bias = decay_bias
        self._t      = 0
        # Moments are lazily initialised in step() on first use.
        self._m = [None] * len(self._params)
        self._v = [None] * len(self._params)

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
        debug = os.environ.get("ORTHON_DEBUG_ADAM") == "1"
        with ortho.no_grad():
            self._t += 1
            bc1 = 1.0 - self._beta1 ** self._t
            bc2 = 1.0 - self._beta2 ** self._t

            for i, p in enumerate(self._params):
                if p.grad is None:
                    continue

                device = p.device

                # Lazy-init moments on the same device as the parameter.
                if self._m[i] is None:
                    self._m[i] = ortho.zeros(list(p.shape)).to(device)
                    self._v[i] = ortho.zeros(list(p.shape)).to(device)

                m, v = self._m[i], self._v[i]

                # First moment: m = beta1*m + (1-beta1)*grad
                m.add_(m, alpha=self._beta1 - 1.0)
                m.add_(p.grad, alpha=1.0 - self._beta1)

                # Second moment: v = beta2*v + (1-beta2)*grad^2
                grad_sq = p.grad * p.grad
                v.add_(v, alpha=self._beta2 - 1.0)
                v.add_(grad_sq, alpha=1.0 - self._beta2)

                # Bias-corrected denominator: sqrt(v/bc2) + eps
                #
                # Both constants are Python scalars. Materialising them as
                # PARAMETER-SHAPED tensors -
                #
                #     shape = list(p.shape)
                #     bc2_tensor = ortho.full(shape, 1.0 / bc2).to(device)
                #     eps_tensor = ortho.full(shape, self._eps).to(device)
                #
                # - would allocate, fill and transfer 2 * p.numel() floats per
                # parameter per step purely to inject two Python floats into
                # broadcasting arithmetic. A scalar operand is rank-0, so the
                # allocation is O(1) regardless of parameter size.
                #
                # The arithmetic is identical: the scalar takes the tensor's
                # own float32 dtype, and `m / denom` composes to
                # mul(m, reciprocal(denom)).
                v_hat = v * (1.0 / bc2)
                sqrt_v_hat = ortho.sqrt(v_hat)
                denom = sqrt_v_hat + self._eps

                # Parameter update: p -= (lr/bc1) * m / denom
                step = m / denom
                if debug:
                    _debug_stats(i, p.shape, p.grad, m, v, v_hat, sqrt_v_hat, denom, step)
                if self._should_decay(p):
                    p.add_(p, alpha=-self._lr * self._weight_decay)
                p.add_(step, alpha=-self._lr / bc1)

    def state_dict(self):
        return {
            "lr": self._lr,
            "betas": (self._beta1, self._beta2),
            "eps": self._eps,
            "weight_decay": self._weight_decay,
            "decay_bias": self._decay_bias,
            "t": self._t,
            "m": self._m,
            "v": self._v,
        }

    def load_state_dict(self, state, map_location=None):
        self._lr = state["lr"]
        self._beta1, self._beta2 = state["betas"]
        self._eps = state["eps"]
        self._weight_decay = state.get("weight_decay", 0.0)
        self._decay_bias = state.get("decay_bias", False)
        self._t = state["t"]
        self._m = self._restore_state_tensors(state["m"], map_location)
        self._v = self._restore_state_tensors(state["v"], map_location)

    def _restore_state_tensors(self, tensors, map_location):
        if len(tensors) != len(self._params):
            raise ValueError(
                "Adam optimizer state parameter count mismatch: "
                f"state {len(tensors)} vs optimizer {len(self._params)}"
            )
        restored = []
        for index, tensor in enumerate(tensors):
            if tensor is None:
                restored.append(None)
                continue
            if list(tensor.shape) != list(self._params[index].shape):
                raise ValueError(
                    "Adam optimizer state shape mismatch for parameter "
                    f"{index}: state {list(tensor.shape)} vs param "
                    f"{list(self._params[index].shape)}"
                )
            target = map_location if map_location is not None else self._params[index].device
            if tensor.device != target:
                tensor = tensor.to(target)
            restored.append(tensor)
        return restored
