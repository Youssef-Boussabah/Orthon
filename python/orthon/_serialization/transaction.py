"""Prepare all state owners before publication; adapt public duck-typed owners."""
import orthon as ortho
from orthon._state import _PreparedLoad, _commit_prepared
from .format import _fail


def _independent(value):
    # Public state only: never deepcopy arbitrary objects or run reducer hooks.
    if isinstance(value, ortho.Tensor):
        with ortho.no_grad():
            return value.to(value.device)
    if isinstance(value, dict):
        return {key: _independent(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_independent(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_independent(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError("checkpoint: generic rollback state contains unsupported value type")


def _generic(owner, state, target, *, model):
    if not hasattr(owner, "state_dict"):
        raise TypeError("checkpoint: generic recipient must expose state_dict() for rollback")
    previous = _independent(owner.state_dict())
    current_devices = [tensor.device for tensor in previous.values()] if model else []
    original_device = (current_devices[0] if current_devices and
                       all(device == current_devices[0] for device in current_devices) else None)
    if model:
        placed = {}
        for name, value in state.items():
            device = target
            if device is None and name in previous:
                device = previous[name].device
            with ortho.no_grad():
                placed[name] = value.to(device) if device is not None else value

        def commit():
            if target is not None and hasattr(owner, "to"):
                owner.to(target)
            owner.load_state_dict(placed)

        def rollback():
            if target is not None and original_device is not None and hasattr(owner, "to"):
                owner.to(original_device)
            owner.load_state_dict(previous)
    else:
        def commit():
            owner.load_state_dict(state, map_location=target)

        def rollback():
            owner.load_state_dict(previous, map_location=None)
    return _PreparedLoad(commit, rollback)


class RestoreTransaction:
    def __init__(self):
        self._participants = []

    def prepare(self, model, optimizer, sections, target, absent):
        # Missing requested sections reject before participant preparation.
        for owner, name in ((model, "model"), (optimizer, "optimizer")):
            if owner is not None:
                if name in absent:
                    _fail("archive has no %s state" % name)
                if not hasattr(owner, "load_state_dict"):
                    raise TypeError("load_checkpoint: %s does not support load_state_dict()" % name)
        placement = lambda parameter: parameter.device
        if model is not None:
            prepare = getattr(model, "_prepare_load_state_dict", None)
            prepared = (prepare(sections["model"], map_location=target) if prepare else
                        _generic(model, sections["model"], target, model=True))
            self._participants.append(prepared)
            placement = prepared.placement or placement
        if optimizer is not None:
            prepare = getattr(optimizer, "_prepare_load_state_dict", None)
            prepared = (prepare(sections["optimizer"], map_location=target,
                                placement=placement) if prepare else
                        _generic(optimizer, sections["optimizer"], target, model=False))
            self._participants.append(prepared)

    def commit(self):
        _commit_prepared(self._participants)
