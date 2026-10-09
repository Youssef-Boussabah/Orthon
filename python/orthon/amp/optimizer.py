"""Explicit mixed precision with stable model owners and FP32 optimization state."""
from dataclasses import dataclass

import orthon as ortho
from orthon._state import _PreparedLoad, _commit_prepared

_LOW_PREC = {"float16", "bfloat16"}
_SUPPORTED = _LOW_PREC | {"float32"}


@dataclass(frozen=True)
class _MixedPrecisionState:
    parameter: ortho.Tensor
    model_dtype: str
    master: ortho.Tensor
    model_device: ortho.Device


def _validate_parameter(parameter):
    if not isinstance(parameter, ortho.Tensor):
        raise TypeError("AMP model parameters must be orthon.Tensor owners")
    if parameter.dtype not in _SUPPORTED:
        raise ValueError("AMP model parameters must be float32/float16/bfloat16")


def _snapshot_master(value, device):
    # to() snapshots even on the same device. Never adopt caller-owned state or
    # a differentiable cast/transfer graph as mutable optimization state.
    with ortho.no_grad():
        master = value.to(device)
        if master.dtype != "float32":
            master = ortho.cast(master, "float32")
    master.requires_grad_(True)
    return master


def _validate_sync(parameter, master):
    # Preflight the whole wrapper before any semantic copy is launched. The
    # semantic operation also enforces these conditions for every native caller.
    if parameter.shape != master.shape or parameter.device != master.device:
        raise ValueError("AMP master shape/device must match model owner")
    if master.dtype != "float32" or parameter.dtype not in _SUPPORTED:
        raise ValueError("AMP synchronization requires an FP32 master")
    if not parameter.is_contiguous or not master.is_contiguous:
        raise RuntimeError("AMP synchronization requires contiguous tensors")


class MixedPrecisionOptimizer:
    """Optimize explicit FP16/BF16 models through FP32 masters.

    FP32 owners pass through. Supported Module transforms retain the Python
    owner and reconcile lazily; explicit parameter replacement does not rebind.
    """

    def __init__(self, model, optimizer_cls, **optimizer_kwargs):
        validate = getattr(model, "_validate_master_gradient_sync", None)
        if validate is not None:
            validate()
        from orthon.distributed import _state
        self._distributed_group = _state.get_active() if _state.is_initialized() else None
        parameters = tuple(model.parameters())
        for parameter in parameters:
            _validate_parameter(parameter)
        self._states = tuple(
            _MixedPrecisionState(p, p.dtype, p if p.dtype == "float32" else
                                 _snapshot_master(p, p.device), p.device)
            for p in parameters
        )
        self._optimizer_factory = optimizer_cls
        self._optimizer_kwargs = dict(optimizer_kwargs)
        self._inner = optimizer_cls((s.master for s in self._states), **optimizer_kwargs)

    def _prepare_states(self):
        prepared, copies = [], []
        changed = False
        for state in self._states:
            p, master = state.parameter, state.master
            _validate_parameter(p)
            if p.shape != master.shape:
                raise ValueError("AMP transformed parameter shape differs from master")
            if p.dtype == "float32":
                if state.model_dtype in _LOW_PREC:
                    if master.device != p.device:
                        master = _snapshot_master(master, p.device)
                    _validate_sync(p, master)
                    copies.append((p, master))
                master = p
            elif state.model_dtype == "float32":
                master = _snapshot_master(p, p.device)
            elif master.device != p.device:
                master = _snapshot_master(master, p.device)
            if p.dtype in _LOW_PREC:
                _validate_sync(p, master)
            if (master is not state.master or p.dtype != state.model_dtype
                    or p.device != state.model_device):
                changed = True
            prepared.append(_MixedPrecisionState(p, p.dtype, master, p.device))
        return tuple(prepared), copies, changed

    def _publish_states(self, states, copies, changed):
        if changed:
            # The bridge prepares all moment pairs before publishing bindings.
            self._inner._rebind_parameters(s.master for s in states)
            with ortho.no_grad():
                for parameter, master in copies:
                    ortho._amp_copy_master_to_model_(parameter, master)
            self._states = states

    def _reconcile(self):
        self._publish_states(*self._prepare_states())

    def zero_grad(self):
        for state in self._states:
            state.parameter.zero_grad()
            if state.master is not state.parameter:
                state.master.zero_grad()

    def _transfer_gradients(self, inv_scale=None):
        states, copies, changed = self._prepare_states()
        gradients = []
        found_inf = False
        current_gradients = tuple(state.parameter.grad for state in states)
        # Validate every current gradient before performing any conversion.
        for state, grad in zip(states, current_gradients):
            if grad is not None:
                if grad.shape != state.parameter.shape or grad.device != state.parameter.device:
                    raise ValueError("AMP gradient shape/device must match model owner")
                if grad.dtype not in _SUPPORTED:
                    raise ValueError("AMP gradients must be float32/float16/bfloat16")
        with ortho.no_grad():
            for state, grad in zip(states, current_gradients):
                if grad is not None:
                    if inv_scale is not None and grad.device.is_cuda():
                        grad, finite = ortho._amp_unscale_to_float32_and_check(grad, inv_scale)
                        found_inf |= not finite
                    else:
                        if state.model_dtype in _LOW_PREC or grad.dtype != "float32":
                            original_dtype = grad.dtype
                            grad = ortho.cast(grad, "float32")
                            if original_dtype == "float32":
                                grad = grad.to(grad.device)
                        if inv_scale is not None:
                            grad = grad * inv_scale
                            found_inf |= not ortho.isfinite_all(grad)
                    if grad.requires_grad or not grad.is_leaf:
                        grad = grad.to(grad.device)
                gradients.append(grad)
        # No model/master/inner/scaler publication occurred during preparation.
        self._publish_states(states, copies, changed)
        for state, grad in zip(self._states, gradients):
            if grad is None:
                state.master.zero_grad()
            else:
                state.master.set_grad(grad)
        return found_inf

    def _unscale_and_check(self, inv_scale):
        return self._transfer_gradients(inv_scale)

    def step(self):
        self._transfer_gradients()
        self._step_prepared()

    def sync_to_model(self):
        self._reconcile()
        with ortho.no_grad():
            for state in self._states:
                if state.master is not state.parameter:
                    ortho._amp_copy_master_to_model_(state.parameter, state.master)

    def _amp_collective_device(self):
        if self._distributed_group is not None:
            self._distributed_group.ensure_open()
        return self._states[0].master.device if self._states else None

    def _all_reduce_master_grads(self):
        from orthon.distributed import _state
        from orthon.distributed._reducer import _reduce_master_gradients
        if self._distributed_group is None and _state.is_initialized():
            self._distributed_group = _state.get_active()
        group = self._distributed_group
        if group is None:
            return
        group.ensure_open()
        if group.world_size > 1:
            _reduce_master_gradients(tuple(s.master for s in self._states), group)

    def _step_prepared(self):
        self._all_reduce_master_grads()
        self._inner.step()
        self.sync_to_model()

    def get_lr(self):
        return self._inner.get_lr()

    def set_lr(self, lr):
        self._inner.set_lr(lr)

    def state_dict(self):
        self._reconcile()
        return {"version": 1,
                "orig_dtypes": [s.model_dtype for s in self._states],
                "masters": [s.master for s in self._states],
                "inner": self._inner.state_dict()}

    def load_state_dict(self, state, map_location=None):
        _commit_prepared([self._prepare_load_state_dict(state, map_location)])

    def _prepare_load_state_dict(self, state, map_location=None, placement=None):
        if type(state.get("version")) is not int or state["version"] != 1:
            raise RuntimeError("MixedPrecisionOptimizer: unsupported state version")
        dtypes, masters, inner_state = list(state["orig_dtypes"]), list(state["masters"]), state["inner"]
        if len(dtypes) != len(self._states) or len(masters) != len(self._states):
            raise ValueError("MixedPrecisionOptimizer state parameter count mismatch")
        target = map_location
        if isinstance(target, str):
            target = ortho.Device(target)
        if target is not None and not isinstance(target, ortho.Device):
            raise TypeError("AMP map_location must be a device string or Device")
        devices = tuple(placement(s.parameter) if placement else s.parameter.device
                        for s in self._states)
        for current, dtype, master, device in zip(self._states, dtypes, masters, devices):
            p = current.parameter
            _validate_parameter(p)
            if dtype != p.dtype:
                raise ValueError("MixedPrecisionOptimizer dtype mismatch")
            if not isinstance(master, ortho.Tensor):
                raise TypeError("AMP saved masters must be orthon.Tensor values")
            if master.shape != p.shape:
                raise ValueError("MixedPrecisionOptimizer master shape mismatch")
            if target is not None and target != device:
                raise ValueError("AMP map_location conflicts with bound model device")
            if p.dtype in _LOW_PREC and not p.is_contiguous and device == p.device:
                raise RuntimeError("AMP synchronization requires contiguous tensors")
        prepared = tuple(
            _MixedPrecisionState(s.parameter, dtype, s.parameter if dtype == "float32"
                                 else _snapshot_master(master, device), device)
            for s, dtype, master, device in zip(self._states, dtypes, masters, devices)
        )
        temporary = self._optimizer_factory((s.master for s in prepared), **self._optimizer_kwargs)
        master_devices = {id(s.master): device for s, device in zip(prepared, devices)}
        inner_prepare = getattr(temporary, "_prepare_load_state_dict", None)
        if inner_prepare is not None:
            inner_prepare(inner_state, map_location=target,
                          placement=lambda p: master_devices[id(p)]).commit()
        else:
            temporary.load_state_dict(inner_state, map_location=target)
        copies, previous_values = [], []
        with ortho.no_grad():
            for s in prepared:
                if s.master is s.parameter:
                    continue
                # Validate against the future representation, without changing
                # the stable owner that Module will publish later.
                parameter = s.parameter
                if parameter.device != s.model_device:
                    parameter = parameter.to(s.model_device)
                _validate_sync(parameter, s.master)
                copies.append((s.parameter, s.master))
                previous_values.append(_snapshot_master(s.parameter, s.parameter.device))
        previous = self._states, self._inner
        attempted = []

        def commit():
            with ortho.no_grad():
                for index, (parameter, master) in enumerate(copies):
                    attempted.append(index)
                    ortho._amp_copy_master_to_model_(parameter, master)
            self._states, self._inner = prepared, temporary

        def rollback():
            self._states, self._inner = previous
            with ortho.no_grad():
                for index in reversed(attempted):
                    parameter, _ = copies[index]
                    value = previous_values[index]
                    # Model rollback follows this participant in reverse order.
                    # Restore current values without trying to undo its move.
                    if value.device != parameter.device:
                        value = value.to(parameter.device)
                    ortho._amp_copy_master_to_model_(parameter, value)
                attempted.clear()

        return _PreparedLoad(commit, rollback)
