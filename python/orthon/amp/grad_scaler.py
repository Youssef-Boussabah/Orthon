"""Finite-aware loss scaling through the mixed-precision wrapper protocol."""
import math
from numbers import Integral, Real

import orthon as ortho


def _real(value, name, predicate):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"GradScaler: {name} must be a finite real number")
    value = float(value)
    if not math.isfinite(value) or not predicate(value):
        raise ValueError(f"GradScaler: invalid {name}")
    return value


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"GradScaler: {name} must be an integer")
    if value < minimum:
        raise ValueError(f"GradScaler: invalid {name}")
    return int(value)


def _boolean(value, name):
    if type(value) is not bool:
        raise TypeError(f"GradScaler: {name} must be bool")
    return value


class GradScaler:
    """One enabled optimizer per update cycle; adaptation occurs in step()."""

    def __init__(self, init_scale=1024.0, growth_factor=2.0, backoff_factor=0.5,
                 growth_interval=2000, enabled=True):
        self._scale = _real(init_scale, "scale", lambda x: x > 0)
        self._growth_factor = _real(growth_factor, "growth_factor", lambda x: x > 1)
        self._backoff_factor = _real(backoff_factor, "backoff_factor", lambda x: 0 < x < 1)
        self._growth_interval = _integer(growth_interval, "growth_interval", 1)
        self._enabled = _boolean(enabled, "enabled")
        self._unscaled = False
        self._found_inf = False
        self._growth_tracker = 0
        self._cycle_optimizer = None

    @property
    def scale_factor(self):
        return self._scale

    @property
    def enabled(self):
        return self._enabled

    @property
    def found_inf(self):
        return self._found_inf

    def scale(self, loss):
        return loss * self._scale if self._enabled else loss

    def _validate_cycle(self, optimizer):
        if self._cycle_optimizer is not None and self._cycle_optimizer is not optimizer:
            raise RuntimeError("GradScaler controls one optimizer per update cycle")
        if self._unscaled and self._cycle_optimizer is None:
            raise RuntimeError("GradScaler restored an unscaled cycle; call update() first")

    def unscale_(self, mp_optimizer):
        if not self._enabled:
            return
        self._validate_cycle(mp_optimizer)
        if self._unscaled:
            return
        found_inf = mp_optimizer._unscale_and_check(1.0 / self._scale)
        # Failed preparation cannot bind the cycle or advance bookkeeping.
        self._cycle_optimizer = mp_optimizer
        self._found_inf = found_inf
        self._unscaled = True

    def _distributed_found_inf(self, mp_optimizer):
        import orthon.distributed as ddist
        if not ddist.is_initialized() or ddist.get_world_size() == 1:
            return self._found_inf
        device = mp_optimizer._amp_collective_device()
        if device is None:
            return self._found_inf
        # The collective needs an addressable [1] FP32 buffer.
        flag = ortho.full([1], 1.0 if self._found_inf else 0.0).to(device)
        ddist.all_reduce(flag, op="sum")
        return float(flag.to("cpu").tolist()[0]) > 0.0

    def step(self, mp_optimizer):
        if not self._enabled:
            mp_optimizer.step()
            return
        self._validate_cycle(mp_optimizer)
        if not self._unscaled:
            self.unscale_(mp_optimizer)
        self._found_inf = self._distributed_found_inf(mp_optimizer)
        if self._found_inf:
            self._scale *= self._backoff_factor
            self._growth_tracker = 0
        else:
            mp_optimizer._step_prepared()
            self._growth_tracker += 1
            if self._growth_tracker >= self._growth_interval:
                self._scale *= self._growth_factor
                self._growth_tracker = 0
        self._unscaled = False

    def update(self):
        self._unscaled = False
        self._found_inf = False
        self._cycle_optimizer = None

    def state_dict(self):
        return {"version": 1, "scale": self._scale,
                "growth_factor": self._growth_factor, "backoff_factor": self._backoff_factor,
                "growth_interval": self._growth_interval, "enabled": self._enabled,
                "unscaled": self._unscaled, "found_inf": self._found_inf,
                "growth_tracker": self._growth_tracker}

    def load_state_dict(self, state):
        if type(state.get("version")) is not int or state["version"] != 1:
            raise RuntimeError("GradScaler: unsupported state version")
        scale = _real(state["scale"], "scale", lambda x: x > 0)
        growth = _real(state["growth_factor"], "growth_factor", lambda x: x > 1)
        backoff = _real(state["backoff_factor"], "backoff_factor", lambda x: 0 < x < 1)
        interval = _integer(state["growth_interval"], "growth_interval", 1)
        enabled = _boolean(state["enabled"], "enabled")
        unscaled = _boolean(state.get("unscaled", False), "unscaled")
        found_inf = _boolean(state.get("found_inf", False), "found_inf")
        tracker = _integer(state.get("growth_tracker", 0), "growth_tracker")
        if tracker >= interval:
            raise ValueError("GradScaler: growth_tracker must be below growth_interval")
        self._scale, self._growth_factor, self._backoff_factor = scale, growth, backoff
        self._growth_interval, self._enabled = interval, enabled
        self._unscaled, self._found_inf, self._growth_tracker = unscaled, found_inf, tracker
        self._cycle_optimizer = None
