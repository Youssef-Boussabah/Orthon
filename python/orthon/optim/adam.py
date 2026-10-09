import os
from dataclasses import dataclass

import orthon as ortho
from orthon._state import _PreparedLoad, _commit_prepared

from ._binding import _Binding, _capture_bindings, _scaled_add_inplace


@dataclass(frozen=True)
class _AdamState:
    m: ortho.Tensor
    v: ortho.Tensor


def _validate_state_profile(parameter, device=None):
    device = parameter.device if device is None else device
    if parameter.dtype not in ("float32", "float64") or (
        device.is_cuda() and parameter.dtype != "float32"
    ):
        raise RuntimeError("bare Adam supports CPU float32/float64 and CUDA float32")


def _prepare_moment(tensor, parameter, device=None):
    """Independent, contiguous, graph-free value snapshot on the bound owner."""
    copied = tensor.to(parameter.device if device is None else device)
    if copied.dtype != parameter.dtype:
        copied = ortho.cast(copied, parameter.dtype)
    return copied


def _prepare_state(state, parameter, device=None):
    if state.m.shape != parameter.shape or state.v.shape != parameter.shape:
        raise ValueError("Adam moment shape does not match bound parameter")
    # Prepare both values before replacing the one per-binding state record.
    return _AdamState(
        _prepare_moment(state.m, parameter, device),
        _prepare_moment(state.v, parameter, device),
    )


def _normalize_location(location):
    if location is None or isinstance(location, ortho.Device):
        return location
    if not isinstance(location, str):
        raise TypeError("Adam map_location must be a device string or Device")
    try:
        return ortho.Device(location)
    except RuntimeError as error:
        raise ValueError(f"invalid Adam map_location: {location!r}") from error


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
        self._bindings = _capture_bindings(params)
        self._lr     = lr
        self._beta1, self._beta2 = betas
        self._eps    = eps
        self._weight_decay = weight_decay
        self._decay_bias = decay_bias
        self._t      = 0

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
        debug = os.environ.get("ORTHON_DEBUG_ADAM") == "1"
        with ortho.no_grad():
            self._t += 1
            bc1 = 1.0 - self._beta1 ** self._t
            bc2 = 1.0 - self._beta2 ** self._t

            for i, binding in enumerate(self._bindings):
                p = binding.parameter
                if p.grad is None:
                    continue

                _validate_state_profile(p)
                state = binding.state
                if state is None:
                    # Allocate the complete lazy pair before publication.
                    state = _AdamState(
                        ortho.zeros(list(p.shape), dtype=p.dtype).to(p.device),
                        ortho.zeros(list(p.shape), dtype=p.dtype).to(p.device),
                    )
                    binding.state = state
                elif any(moment.shape != p.shape or moment.dtype != p.dtype
                         or moment.device != p.device for moment in (state.m, state.v)):
                    state = _prepare_state(state, p)
                    binding.state = state

                m, v = state.m, state.v

                # First moment: m = beta1*m + (1-beta1)*grad
                _scaled_add_inplace(m, m, self._beta1 - 1.0)
                _scaled_add_inplace(m, p.grad, 1.0 - self._beta1)

                # Second moment: v = beta2*v + (1-beta2)*grad^2
                grad_sq = p.grad * p.grad
                _scaled_add_inplace(v, v, self._beta2 - 1.0)
                _scaled_add_inplace(v, grad_sq, 1.0 - self._beta2)

                # Rank-0 scalars preserve dtype and use O(1) storage. Keep the Adam
                # denominator sqrt(v / bc2) + eps, with epsilon outside the square root.
                v_hat = v * (1.0 / bc2)
                sqrt_v_hat = ortho.sqrt(v_hat)
                denom = sqrt_v_hat + self._eps

                # Parameter update: p -= (lr/bc1) * m / denom
                step = m / denom
                if debug:
                    _debug_stats(i, p.shape, p.grad, m, v, v_hat, sqrt_v_hat, denom, step)
                if self._should_decay(p):
                    _scaled_add_inplace(p, p, -self._lr * self._weight_decay)
                _scaled_add_inplace(p, step, -self._lr / bc1)

    def state_dict(self):
        return {
            "lr": self._lr,
            "betas": (self._beta1, self._beta2),
            "eps": self._eps,
            "weight_decay": self._weight_decay,
            "decay_bias": self._decay_bias,
            "t": self._t,
            "m": [None if b.state is None else b.state.m for b in self._bindings],
            "v": [None if b.state is None else b.state.v for b in self._bindings],
        }

    def load_state_dict(self, state, map_location=None):
        _commit_prepared([self._prepare_load_state_dict(state, map_location)])

    def _prepare_load_state_dict(self, state, map_location=None, placement=None):
        lr = state["lr"]
        beta1, beta2 = state["betas"]
        eps = state["eps"]
        weight_decay = state.get("weight_decay", 0.0)
        decay_bias = state.get("decay_bias", False)
        t = state["t"]
        moments1, moments2 = list(state["m"]), list(state["v"])
        target = _normalize_location(map_location)
        devices = tuple(placement(b.parameter) if placement else b.parameter.device
                        for b in self._bindings)
        if len(moments1) != len(self._bindings) or len(moments2) != len(self._bindings):
            raise ValueError("Adam optimizer state parameter count mismatch")

        # Validate the complete input before copying any Tensor, then prepare
        # independently owned pairs locally. Nothing is published on failure.
        for binding, m, v, device in zip(self._bindings, moments1, moments2, devices):
            p = binding.parameter
            if target is not None and target != device:
                raise ValueError("Adam map_location conflicts with bound parameter device")
            if (m is None) != (v is None):
                raise ValueError("Adam moments must be initialized together")
            if m is None:
                continue
            if not isinstance(m, ortho.Tensor) or not isinstance(v, ortho.Tensor):
                raise TypeError("Adam moments must be orthon.Tensor values or both None")
            if m.shape != p.shape or v.shape != p.shape:
                raise ValueError("Adam moment shape does not match bound parameter")
            _validate_state_profile(p, device)

        with ortho.no_grad():
            prepared = tuple(
                _Binding(binding.parameter, None if m is None else
                         _prepare_state(_AdamState(m, v), binding.parameter, device))
                for binding, m, v, device in zip(self._bindings, moments1, moments2, devices)
            )
        previous = (self._lr, self._beta1, self._beta2, self._eps,
                    self._weight_decay, self._decay_bias, self._t, self._bindings)

        def commit():
            (self._lr, self._beta1, self._beta2, self._eps, self._weight_decay,
             self._decay_bias, self._t, self._bindings) = (
                lr, beta1, beta2, eps, weight_decay, decay_bias, t, prepared)

        def rollback():
            (self._lr, self._beta1, self._beta2, self._eps, self._weight_decay,
             self._decay_bias, self._t, self._bindings) = previous

        return _PreparedLoad(commit, rollback)

    def _rebind_parameters(self, params):
        """Private AMP bridge: preserve complete positional history atomically."""
        captured = _capture_bindings(params)
        if len(captured) != len(self._bindings):
            raise ValueError("Adam rebind parameter count mismatch")
        for old, new in zip(self._bindings, captured):
            if old.parameter.shape != new.parameter.shape:
                raise ValueError("Adam rebind parameter shape mismatch")
            if old.state is not None:
                _validate_state_profile(new.parameter)
        with ortho.no_grad():
            prepared = tuple(
                _Binding(new.parameter, None if old.state is None else
                         _prepare_state(old.state, new.parameter))
                for old, new in zip(self._bindings, captured)
            )
        self._bindings = prepared
