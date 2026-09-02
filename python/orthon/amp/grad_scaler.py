import orthon as ortho

_LOW_PREC = {"float16", "bfloat16"}


class GradScaler:
    """Dynamic finite-aware loss scaler for mixed-precision training."""

    def __init__(
        self,
        init_scale: float = 1024.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
    ):
        if init_scale <= 0.0:
            raise ValueError("init_scale must be positive")
        if growth_factor <= 1.0:
            raise ValueError("growth_factor must be greater than 1.0")
        if backoff_factor <= 0.0 or backoff_factor >= 1.0:
            raise ValueError("backoff_factor must be in (0.0, 1.0)")
        if growth_interval < 1:
            raise ValueError("growth_interval must be at least 1")
        self._scale = float(init_scale)
        self._growth_factor = float(growth_factor)
        self._backoff_factor = float(backoff_factor)
        self._growth_interval = int(growth_interval)
        self._enabled = enabled
        self._unscaled = False
        self._found_inf = False
        self._growth_tracker = 0

    @property
    def scale_factor(self) -> float:
        return self._scale

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def found_inf(self) -> bool:
        return self._found_inf

    def scale(self, loss):
        """Return loss * scale_factor (autograd-safe). No-op when disabled."""
        if not self._enabled:
            return loss
        # A Python scalar operand is rank-0 and takes the tensor's exact
        # Device and dtype by contract, so this needs no loss-shaped constant
        # tensor to be built, moved and cast first.
        return loss * self._scale

    def unscale_(self, mp_optimizer):
        """Divide FP32 master gradients by scale_factor and record inf/nan state."""
        if not self._enabled or self._unscaled:
            return
        inv = 1.0 / self._scale
        found_inf = False
        fused_unscale = getattr(ortho, "_amp_unscale_to_float32_and_check", None)
        with ortho.no_grad():
            for sub_mod, name, orig_dtype, master in mp_optimizer._entries:
                if orig_dtype in _LOW_PREC:
                    p = sub_mod._parameters[name]
                    if p.grad is None:
                        continue
                    if fused_unscale is not None and p.grad.device.is_cuda():
                        fp32_grad, is_finite = fused_unscale(p.grad, inv)
                        master.set_grad(fp32_grad)
                        if not is_finite:
                            found_inf = True
                        continue
                    fp32_grad = ortho.cast(p.grad, "float32")
                    # `inv` is a Python scalar, not a gradient-shaped
                    # constant tensor. This is the fallback path; the fused
                    # CUDA unscale above is used where available.
                    master.set_grad(fp32_grad * inv)
                    if not ortho.isfinite_all(master.grad):
                        found_inf = True
                else:
                    if master.grad is None:
                        continue
                    if (
                        fused_unscale is not None
                        and master.grad.device.is_cuda()
                        and master.grad.dtype == "float32"
                    ):
                        fp32_grad, is_finite = fused_unscale(master.grad, inv)
                        master.set_grad(fp32_grad)
                        if not is_finite:
                            found_inf = True
                        continue
                    master.set_grad(master.grad * inv)
                    if not ortho.isfinite_all(master.grad):
                        found_inf = True
        self._unscaled = True
        self._found_inf = found_inf

    def _distributed_found_inf(self, mp_optimizer) -> bool:
        try:
            import orthon.distributed as ddist
        except Exception:
            return self._found_inf
        if not ddist.is_initialized() or ddist.get_world_size() == 1:
            return self._found_inf

        device = None
        for _, _, _, master in mp_optimizer._entries:
            device = master.device
            break
        if device is None:
            return self._found_inf

        # Deliberately NOT a Python scalar: this [1]-element tensor is the
        # operand of a collective. all_reduce needs a real, addressable buffer,
        # and its shape is part of the distributed contract.
        flag = ortho.full([1], 1.0 if self._found_inf else 0.0)
        if device.is_cuda():
            flag = flag.to(device)
        ddist.all_reduce(flag, op="sum")
        return float(flag.to("cpu").tolist()[0]) > 0.0

    def step(self, mp_optimizer):
        """Unscale, skip non-finite grads, otherwise step and sync to model."""
        if not self._enabled:
            mp_optimizer.step()
            return
        if not self._unscaled:
            self.unscale_(mp_optimizer)
        self._found_inf = self._distributed_found_inf(mp_optimizer)
        if self._found_inf:
            self._scale *= self._backoff_factor
            self._growth_tracker = 0
            self._unscaled = False
            return
        if hasattr(mp_optimizer, "_all_reduce_master_grads"):
            mp_optimizer._all_reduce_master_grads()
        mp_optimizer._inner.step()
        mp_optimizer.sync_to_model()
        self._growth_tracker += 1
        if self._growth_tracker >= self._growth_interval:
            self._scale *= self._growth_factor
            self._growth_tracker = 0
        self._unscaled = False

    def update(self):
        """Reset per-step finite-check state."""
        self._unscaled = False
        self._found_inf = False

    def state_dict(self):
        return {
            "version": 1,
            "scale": self._scale,
            "growth_factor": self._growth_factor,
            "backoff_factor": self._backoff_factor,
            "growth_interval": self._growth_interval,
            "enabled": self._enabled,
            "unscaled": self._unscaled,
            "found_inf": self._found_inf,
            "growth_tracker": self._growth_tracker,
        }

    def load_state_dict(self, state):
        if state.get("version") != 1:
            raise RuntimeError("GradScaler: unsupported state version")
        scale = float(state["scale"])
        growth_factor = float(state["growth_factor"])
        backoff_factor = float(state["backoff_factor"])
        growth_interval = int(state["growth_interval"])
        if scale <= 0.0:
            raise ValueError("GradScaler: state scale must be positive")
        if growth_factor <= 1.0:
            raise ValueError("GradScaler: state growth_factor must be greater than 1.0")
        if backoff_factor <= 0.0 or backoff_factor >= 1.0:
            raise ValueError("GradScaler: state backoff_factor must be in (0.0, 1.0)")
        if growth_interval < 1:
            raise ValueError("GradScaler: state growth_interval must be at least 1")
        self._scale = scale
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._enabled = bool(state["enabled"])
        self._unscaled = bool(state.get("unscaled", False))
        self._found_inf = bool(state.get("found_inf", False))
        self._growth_tracker = int(state.get("growth_tracker", 0))
