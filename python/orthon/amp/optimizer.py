import orthon as ortho

_LOW_PREC = {"float16", "bfloat16"}


def _iter_params_with_module(model):
    """Yield (submodule, param_name, param_tensor) for every parameter, depth-first."""
    seen = set()

    def _walk(m):
        for name, p in m._parameters.items():
            pid = id(p)
            if pid not in seen:
                seen.add(pid)
                yield m, name, p
        for child in m._modules.values():
            yield from _walk(child)

    yield from _walk(model)


class MixedPrecisionOptimizer:
    """Maintains FP32 master weights for a FP16/BF16 model.

    For each FP16/BF16 parameter a FP32 master copy is created and handed to
    the inner optimizer.  Before each step the low-precision gradient is cast
    to FP32 and written onto the master; after the step the master is cast
    back to the original dtype and written into the model.

    FP32 parameters pass through unchanged; the master is the model param.

    Usage::
        model.to("cuda").half()
        opt = MixedPrecisionOptimizer(model, Adam, lr=1e-3)
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()
    """

    def __init__(self, model, optimizer_cls, **optimizer_kwargs):
        # (submodule, param_name, orig_dtype, master_tensor)
        self._entries = []
        masters = []

        for sub_mod, name, param in _iter_params_with_module(model):
            orig_dtype = param.dtype
            if orig_dtype in _LOW_PREC:
                # Master creation is a state snapshot, not a graph operation:
                # cast() is differentiable, so the conversion runs under an
                # explicit no_grad to keep the master a detached leaf.
                with ortho.no_grad():
                    master = ortho.cast(param, "float32")
                master.requires_grad_(True)
            else:
                master = param          # FP32: master IS the model param
            self._entries.append((sub_mod, name, orig_dtype, master))
            masters.append(master)

        self._inner = optimizer_cls(masters, **optimizer_kwargs)

    # Public API.

    def zero_grad(self):
        """Zero gradients on model params (backward fills these) and master params."""
        for sub_mod, name, orig_dtype, master in self._entries:
            p = sub_mod._parameters[name]
            if p.grad is not None:
                p.zero_grad()
        self._inner.zero_grad()

    def step(self):
        """Cast low-precision grads -> FP32 masters, step, then sync back to model."""
        with ortho.no_grad():
            for sub_mod, name, orig_dtype, master in self._entries:
                if orig_dtype in _LOW_PREC:
                    p = sub_mod._parameters[name]
                    if p.grad is not None:
                        master.set_grad(ortho.cast(p.grad, "float32"))

        self._inner.step()
        self.sync_to_model()

    def sync_to_model(self):
        """Cast FP32 masters back to original dtype and update model parameters."""
        copy_master_to_model = getattr(ortho, "_amp_copy_master_to_model_", None)
        with ortho.no_grad():
            for sub_mod, name, orig_dtype, master in self._entries:
                if orig_dtype in _LOW_PREC:
                    param = sub_mod._parameters[name]
                    if (
                        copy_master_to_model is not None
                        and param.device.is_cuda()
                        and master.device.is_cuda()
                    ):
                        copy_master_to_model(param, master)
                        continue
                    restored = ortho.cast(master, orig_dtype)
                    restored.requires_grad_(True)
                    sub_mod._parameters[name] = restored

    def _all_reduce_master_grads(self):
        import orthon.distributed as ddist

        if not ddist.is_initialized() or ddist.get_world_size() == 1:
            return
        scale = 1.0 / float(ddist.get_world_size())
        with ortho.no_grad():
            for _, _, _, master in self._entries:
                if master.grad is None:
                    continue
                if master.grad.dtype != "float32":
                    raise RuntimeError(
                        "AMP DDP only synchronizes FP32 master gradients; "
                        f"got {master.grad.dtype}"
                    )
                ddist.all_reduce(master.grad, op="sum")
                master.grad.mul_(scale)

    def get_lr(self):
        return self._inner.get_lr()

    def set_lr(self, lr):
        self._inner.set_lr(lr)

    def state_dict(self):
        return {
            "version": 1,
            "orig_dtypes": [orig_dtype for _, _, orig_dtype, _ in self._entries],
            "masters": [master for _, _, _, master in self._entries],
            "inner": self._inner.state_dict(),
        }

    def load_state_dict(self, state, map_location=None):
        if state.get("version") != 1:
            raise RuntimeError("MixedPrecisionOptimizer: unsupported state version")
        orig_dtypes = list(state["orig_dtypes"])
        saved_masters = list(state["masters"])
        if len(orig_dtypes) != len(self._entries) or len(saved_masters) != len(self._entries):
            raise ValueError(
                "MixedPrecisionOptimizer state parameter count mismatch: "
                f"state {len(saved_masters)} vs optimizer {len(self._entries)}"
            )

        restored_entries = []
        restored_masters = []
        for index, ((sub_mod, name, orig_dtype, _), saved_dtype, saved_master) in enumerate(
            zip(self._entries, orig_dtypes, saved_masters)
        ):
            if saved_dtype != orig_dtype:
                raise ValueError(
                    "MixedPrecisionOptimizer dtype mismatch for parameter "
                    f"{index}: state {saved_dtype} vs optimizer {orig_dtype}"
                )
            param = sub_mod._parameters[name]
            if list(saved_master.shape) != list(param.shape):
                raise ValueError(
                    "MixedPrecisionOptimizer master shape mismatch for parameter "
                    f"{index}: state {list(saved_master.shape)} vs param {list(param.shape)}"
                )
            if orig_dtype in _LOW_PREC:
                target = map_location if map_location is not None else param.device
                # Saved masters require grad, so device/dtype reconciliation
                # must be an explicit no_grad snapshot - the restored
                # master has to be a detached leaf, not a ToNode/CastNode
                # output hanging off the saved tensor.
                master = saved_master
                with ortho.no_grad():
                    if master.device != target:
                        master = master.to(target)
                    if master.dtype != "float32":
                        master = ortho.cast(master, "float32")
                master.requires_grad_(True)
            else:
                master = param
            restored_entries.append((sub_mod, name, orig_dtype, master))
            restored_masters.append(master)

        self._entries = restored_entries
        if hasattr(self._inner, "_params"):
            self._inner._params = restored_masters
        self._inner.load_state_dict(state["inner"], map_location=map_location)
        self.sync_to_model()
