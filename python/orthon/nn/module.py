import orthon as ortho
from orthon._state import _PreparedLoad, _commit_prepared


# Dtypes that Module.cast()/half()/bfloat16() will convert a buffer to. Buffers
# holding integer or boolean data (index tables, counters, masks) keep their own
# dtype so that a model-wide half() call cannot corrupt them. Parameters are
# always cast, matching the pre-existing behaviour.
_FLOATING_DTYPES = ("float16", "bfloat16", "float32", "float64")


class Parameter:
    """Wraps a Tensor to mark it as a trainable parameter."""

    def __init__(self, data):
        if not isinstance(data, ortho.Tensor):
            raise TypeError(f"Parameter expects a Tensor, got {type(data).__name__}")
        data.requires_grad_(True)
        self._data = data


class Module:
    """Owns explicit parameter, buffer and child registration and model state."""

    def __init__(self):
        object.__setattr__(self, '_parameters', {})
        object.__setattr__(self, '_buffers', {})
        object.__setattr__(self, '_non_persistent_buffers', set())
        object.__setattr__(self, '_modules', {})
        object.__setattr__(self, 'training', True)

    # registration: validate first, then publish one local meaning
    @staticmethod
    def _check_buffer_value(name, tensor):
        if tensor is not None and not isinstance(tensor, ortho.Tensor):
            raise TypeError(
                f"buffer '{name}' expects a Tensor or None, "
                f"got {type(tensor).__name__}")

    def _set_member(self, name, value, kind, *, persistent=True):
        """Update local registration atomically.

        Parameter/child assignment may replace an ordinary attribute or each other.
        Buffer roles persist until re-registration; reject role changes before mutation.
        Sequential permits numeric child names."""
        params = self._parameters
        buffers = self._buffers
        modules = self._modules
        attributes = object.__getattribute__(self, '__dict__')
        if kind != 'attribute':
            if not isinstance(name, str):
                raise TypeError(f"{kind} name must be a string")
            if not name or '.' in name or not (
                name.isidentifier() or (kind == 'module' and name.isdecimal())
            ):
                raise ValueError(f"invalid {kind} name {name!r}")
            if hasattr(type(self), name) or name in (
                '_parameters', '_buffers', '_modules',
                '_non_persistent_buffers', 'training',
            ):
                raise KeyError(f"'{name}' is already an attribute of this module")
        if kind == 'buffer':
            if name in params or name in modules or name in attributes:
                raise KeyError(f"'{name}' is already an attribute or registered member")
            self._check_buffer_value(name, value)
            # Keep the established truth-value policy; evaluate before writes.
            persistent = bool(persistent)
        elif name in buffers:
            raise KeyError(f"cannot assign a {kind} to registered buffer '{name}'")
        elif kind == 'parameter':
            if not isinstance(value, ortho.Tensor):
                raise TypeError("registered parameter must be a Tensor")
        elif kind == 'module':
            if not isinstance(value, Module):
                raise TypeError("registered child must be a Module")

        if kind == 'attribute':
            # A descriptor can reject assignment. Let it do so before cleanup.
            object.__setattr__(self, name, value)
        else:
            attributes.pop(name, None)
        if kind != 'parameter':
            params.pop(name, None)
        if kind != 'module':
            modules.pop(name, None)
        if kind != 'buffer':
            buffers.pop(name, None)
            self._non_persistent_buffers.discard(name)
        if kind == 'parameter':
            params[name] = value
        elif kind == 'module':
            modules[name] = value
        elif kind == 'buffer':
            buffers[name] = value
            if persistent:
                self._non_persistent_buffers.discard(name)
            else:
                self._non_persistent_buffers.add(name)

    def _register_module(self, name, module):
        self._set_member(name, module, 'module')

    def register_buffer(self, name, tensor, persistent=True):
        """Register Tensor/None state; nonpersistent slots transform but do not save.

        Reassignment keeps the slot's persistence. None remains registered but
        is omitted from tensor enumeration and state_dict.
        """
        self._set_member(name, tensor, 'buffer', persistent=persistent)

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._set_member(name, value._data, 'parameter')
        elif isinstance(value, Module):
            self._register_module(name, value)
        elif name in self._buffers:
            self._set_member(name, value, 'buffer',
                             persistent=name not in self._non_persistent_buffers)
        else:
            self._set_member(name, value, 'attribute')

    def __getattr__(self, name):
        params = object.__getattribute__(self, '_parameters')
        if name in params:
            return params[name]
        buffers = object.__getattribute__(self, '_buffers')
        if name in buffers:
            return buffers[name]
        modules = object.__getattribute__(self, '_modules')
        if name in modules:
            return modules[name]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    # traversal
    def _walk_modules(self, prefix='', *, recurse=True, unique=False):
        """Preorder paths, or unique Module owners for state transformations.

        Path traversal preserves shared-child aliases. Ancestors, rather than
        a global visited set, stop cycles without discarding sibling paths.
        This is defensive traversal, not support for cyclic model graphs.
        """
        seen = set()

        def walk(module, path, ancestors):
            identity = id(module)
            if identity in ancestors or (unique and identity in seen):
                return
            seen.add(identity)
            yield path, module
            if recurse:
                for name, child in module._modules.items():
                    yield from walk(child, path + name + '.', ancestors | {identity})

        yield from walk(self, prefix, set())

    def _state_entries(self, _prefix='', *, parameters=True, buffers=True,
                       persistent_only=True, include_none=False, recurse=True):
        """Yield (key, owner, kind, local_name, tensor) in registration order.

        Local parameters precede buffers, then children. Values/path aliases
        are deliberately retained; unnamed discovery alone deduplicates them.
        Default flags are the persistent state_dict/load_state_dict protocol.
        """
        for prefix, module in self._walk_modules(_prefix, recurse=recurse):
            if parameters:
                for name, tensor in module._parameters.items():
                    yield prefix + name, module, 'parameter', name, tensor
            if buffers:
                for name, tensor in module._buffers.items():
                    if persistent_only and name in module._non_persistent_buffers:
                        continue
                    if tensor is not None or include_none:
                        yield prefix + name, module, 'buffer', name, tensor

    def parameters(self):
        """Yield unique registered Tensor owners, including frozen parameters."""
        seen = set()
        for _, tensor in self.named_parameters():
            if id(tensor) not in seen:
                seen.add(id(tensor))
                yield tensor

    def named_parameters(self, prefix=""):
        """Yield every registered parameter path, including tied aliases."""
        for key, _, _, _, tensor in self._state_entries(prefix, buffers=False):
            yield key, tensor

    def named_buffers(self, prefix="", recurse=True):
        """Yield every Tensor buffer path, including nonpersistent aliases."""
        for key, _, _, _, tensor in self._state_entries(
            prefix, parameters=False, persistent_only=False, recurse=recurse,
        ):
            yield key, tensor

    def buffers(self, recurse=True):
        """Yield unique buffer Tensor owners; omit None slots."""
        seen = set()
        for _, tensor in self.named_buffers(recurse=recurse):
            if id(tensor) not in seen:
                seen.add(id(tensor))
                yield tensor

    # training mode
    def _set_training(self, training):
        for _, module in self._walk_modules(unique=True):
            module.training = training
        return self

    def train(self):
        """Recursively set training mode. Returns self."""
        return self._set_training(True)

    def eval(self):
        """Recursively set evaluation mode. Returns self."""
        return self._set_training(False)

    def zero_grad(self):
        """Zero out gradients for all parameters."""
        for p in self.parameters():
            if p.grad is not None:
                p.zero_grad()

    # device and dtype
    def _apply_tensor_transform(self, transform, *, floating_buffers_only=False):
        """Prepare each unique owner once, then publish graph-free model state.

        Parameter Python owners survive; buffer-only owners are replaced as
        before. Buffer ties (including aliases of parameters) remain tied.
        Preparation completes before any owner/registration mutation, so an
        unsupported later conversion leaves the whole model unchanged.
        """
        modules = [m for _, m in self._walk_modules(unique=True)]
        parameter_ids = {id(p) for m in modules for p in m._parameters.values()}
        owners = {}
        buffer_slots = []
        for module in modules:
            for p in module._parameters.values():
                owners.setdefault(id(p), p)
            for name, b in module._buffers.items():
                if b is None or (floating_buffers_only and
                                 b.dtype not in _FLOATING_DTYPES and
                                 id(b) not in parameter_ids):
                    continue
                owners.setdefault(id(b), b)
                buffer_slots.append((module, name, b))
        prepared = {}
        with ortho.no_grad():
            for identity, tensor in owners.items():
                transformed = transform(tensor)
                if transformed.shape != tensor.shape:
                    raise RuntimeError("model transform must preserve shape")
                # Same-dtype cast may return the original nonleaf value.
                # Registered model transformations must publish leaves.
                if not transformed.is_leaf:
                    transformed = transformed.to(transformed.device)
                transformed.requires_grad_(tensor.requires_grad)
                prepared[identity] = transformed
            for identity in parameter_ids:
                owners[identity]._rebind_model_value_(prepared[identity])
                prepared[identity] = owners[identity]
            for module, name, original in buffer_slots:
                module._set_member(
                    name, prepared[id(original)], 'buffer',
                    persistent=name not in module._non_persistent_buffers,
                )
        return self

    def to(self, device):
        """Move registered state to the exact device, preserving parameter owners."""
        return self._apply_tensor_transform(lambda tensor: tensor.to(device))

    def cast(self, dtype):
        """Cast parameters and floating buffers; preserve parameter owners."""
        return self._apply_tensor_transform(
            lambda tensor: ortho.cast(tensor, dtype), floating_buffers_only=True,
        )

    def half(self):
        """Cast to float16. Returns self."""
        return self.cast("float16")

    def bfloat16(self):
        """Cast to bfloat16. Returns self."""
        return self.cast("bfloat16")

    # persistent state
    def state_dict(self, _prefix=''):
        """Return live parameter and persistent-buffer tensors by name.

        Values alias current model state. File save APIs instead snapshot values,
        so later mutation cannot change an archive."""
        return {key: tensor for key, _, _, _, tensor in self._state_entries(_prefix)}

    def load_state_dict(self, state, strict=True):
        """Copy values through the same prepared state-owner protocol as files."""
        _commit_prepared([self._prepare_load_state_dict(state, strict=strict)])
        return self

    def _prepare_load_state_dict(self, state, strict=True, map_location=None):
        """Validate/convert all values and placement without publishing state."""
        entries = list(self._state_entries())
        current_keys = {key for key, _, _, _, _ in entries}
        if strict:
            missing    = current_keys - set(state.keys())
            unexpected = set(state.keys()) - current_keys
            if missing:
                raise KeyError(f"Missing keys in state_dict: {sorted(missing)}")
            if unexpected:
                raise KeyError(f"Unexpected keys in state_dict: {sorted(unexpected)}")

        # Prepare independent destination-device/dtype values before publication.
        # Snapshot self-loads to prevent aliases from being overwritten mid-copy;
        # a later invalid entry must leave every owner unchanged.
        prepared = []
        tied = {}
        for key, owner, kind, local_name, current in entries:
            if key not in state:
                continue
            src = state[key]
            if not isinstance(src, ortho.Tensor):
                raise TypeError(
                    f"state_dict entry '{key}' must be a Tensor, "
                    f"got {type(src).__name__}")
            if list(current.shape) != list(src.shape):
                raise ValueError(
                    f"size mismatch for '{key}': "
                    f"current {list(current.shape)} vs loaded {list(src.shape)}"
                )

            # Canonical to() always makes an independent logical copy, even on
            # the same device, and materializes a strided source. Explicit
            # no_grad keeps source computation history out of model state.
            with ortho.no_grad():
                restored = src.to(map_location if map_location is not None else current.device)
                if restored.dtype != current.dtype:
                    restored = ortho.cast(restored, current.dtype)
            if kind == 'parameter':
                identity = id(current)
                if identity in tied:
                    previous_key, previous = tied[identity]
                    # Numerical consistency, including NaNs. Destination
                    # conversion occurs first, so comparison uses loaded values.
                    first, second = previous.tolist(), restored.tolist()
                    if not all(a == b or (a != a and b != b)
                               for a, b in zip(first, second)):
                        raise ValueError(
                            f"conflicting tied parameter state: '{previous_key}' and '{key}'")
                    continue
                tied[identity] = (key, restored)
            prepared.append((owner, kind, local_name, current, restored))

        # Explicit placement also transforms nonpersistent registered buffers.
        # Incoming parameter values themselves are the prepared representation.
        moves = []
        parameter_ids = {id(p) for p in self.parameters()}
        if map_location is not None:
            loaded_buffers = {(id(owner), name) for owner, kind, name, _, _ in prepared
                              if kind == 'buffer'}
            transformed = {}
            for _, owner in self._walk_modules(unique=True):
                for name, tensor in owner._buffers.items():
                    if tensor is None or tensor.device == map_location:
                        continue
                    if (id(owner), name) in loaded_buffers:
                        continue
                    if id(tensor) in parameter_ids:
                        replacement = tensor
                    else:
                        if id(tensor) not in transformed:
                            with ortho.no_grad():
                                value = tensor.to(map_location)
                                value.requires_grad_(tensor.requires_grad)
                            transformed[id(tensor)] = value
                        replacement = transformed[id(tensor)]
                    moves.append((owner, name, tensor, replacement))
        originals = {}
        with ortho.no_grad():
            for _, kind, _, current, _ in prepared:
                if kind != 'parameter':
                    continue
                moved = map_location is not None and current.device != map_location
                # Retain the exact old implementation, including strided layout,
                # gradients and hooks, while the registered facade moves.
                representation = current._model_value_reference() if moved else None
                originals[id(current)] = (current.to(current.device), representation)
            for _, kind, _, current, restored in prepared:
                if kind == 'buffer' or (map_location is not None and current.device != map_location):
                    restored.requires_grad_(current.requires_grad)

        # Stage 2: prepared commit and rollback callbacks.
        attempted = []

        def commit():
            with ortho.no_grad():
                for owner, kind, name, current, restored in prepared:
                    attempted.append((owner, kind, name, current))
                    if kind == 'buffer':
                        owner._set_member(name, restored, 'buffer',
                                          persistent=name not in owner._non_persistent_buffers)
                    elif map_location is not None and current.device != map_location:
                        current._rebind_model_value_(restored)
                    else:
                        self._copy_state_value(current, restored)
                for owner, name, current, replacement in moves:
                    attempted.append((owner, 'buffer', name, current))
                    owner._set_member(name, replacement, 'buffer',
                                      persistent=name not in owner._non_persistent_buffers)

        def rollback():
            with ortho.no_grad():
                for owner, kind, name, current in reversed(attempted):
                    if kind == 'buffer':
                        owner._set_member(name, current, 'buffer',
                                          persistent=name not in owner._non_persistent_buffers)
                    else:
                        values, representation = originals[id(current)]
                        if representation is not None:
                            current._rebind_model_value_(representation)
                        self._copy_state_value(current, values)
                attempted.clear()

        def placement(parameter):
            return (map_location if map_location is not None and id(parameter) in parameter_ids
                    else parameter.device)

        return _PreparedLoad(commit, rollback, placement)

    @staticmethod
    def _copy_state_value(destination, source):
        """Copy prepared values while retaining destination ownership and layout.

        Partition strided destinations into contiguous subviews. This preserves
        Buffer aliases and autograd metadata without replacing the owner."""
        if destination.is_contiguous:
            destination._copy_data_(source.contiguous())
        else:
            for index in range(int(destination.shape[0])):
                Module._copy_state_value(destination[index], source[index])

    # call protocol
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} must implement forward()")

    def __repr__(self):
        name = type(self).__name__
        if not self._modules and not self._parameters and not self._buffers:
            return f"{name}()"
        lines = [f"{name}("]
        for k, m in self._modules.items():
            lines.append(f"  ({k}): {repr(m)}")
        for k, p in self._parameters.items():
            lines.append(f"  ({k}): Parameter(shape={p.shape})")
        for k, b in self._buffers.items():
            shape = None if b is None else b.shape
            lines.append(f"  ({k}): Buffer(shape={shape})")
        lines.append(")")
        return "\n".join(lines)
