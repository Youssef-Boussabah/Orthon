import orthon as ortho


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
    """Base class for all neural network modules, mirroring PyTorch's nn.Module."""

    def __init__(self):
        object.__setattr__(self, '_parameters', {})
        object.__setattr__(self, '_buffers', {})
        object.__setattr__(self, '_non_persistent_buffers', set())
        object.__setattr__(self, '_modules', {})
        object.__setattr__(self, 'training', True)

    # ---- buffer registration ----

    def register_buffer(self, name, tensor, persistent=True):
        """Register a non-trainable Tensor as module state.

        Buffers move with .to() and cast with .cast(), are returned by
        buffers()/named_buffers(), and are never returned by parameters().
        A persistent buffer appears in state_dict(); a non-persistent one does
        not, but still follows the module across devices and dtypes.

        `tensor` may be None to declare the slot without materialising it yet;
        a None buffer is skipped by enumeration, state_dict, .to() and .cast()
        until it is assigned a real Tensor.
        """
        params = object.__getattribute__(self, '_parameters')
        buffers = object.__getattribute__(self, '_buffers')
        modules = object.__getattribute__(self, '_modules')
        non_persistent = object.__getattribute__(self, '_non_persistent_buffers')

        if not isinstance(name, str):
            raise TypeError(
                f"buffer name must be a string, got {type(name).__name__}")
        if name == '':
            raise ValueError("buffer name must not be empty")
        if '.' in name:
            raise ValueError(
                f"buffer name must not contain '.' (got {name!r}); "
                "'.' separates levels in state_dict keys")
        if not name.isidentifier():
            raise ValueError(f"buffer name {name!r} is not a valid attribute name")

        if name in params:
            raise KeyError(f"'{name}' is already registered as a parameter")
        if name in modules:
            raise KeyError(f"'{name}' is already registered as a child module")
        if name not in buffers:
            # Anything already reachable by attribute lookup would shadow the
            # buffer, leaving it registered but unreadable as self.<name>.
            if name in object.__getattribute__(self, '__dict__'):
                raise KeyError(f"'{name}' is already an attribute of this module")
            if hasattr(type(self), name):
                raise KeyError(
                    f"'{name}' is already an attribute of "
                    f"{type(self).__name__}")

        self._check_buffer_value(name, tensor)

        buffers[name] = tensor
        if persistent:
            non_persistent.discard(name)
        else:
            non_persistent.add(name)

    @staticmethod
    def _check_buffer_value(name, tensor):
        if tensor is None:
            return
        if isinstance(tensor, Parameter):
            raise TypeError(
                f"buffer '{name}' cannot be a Parameter; buffers are not trainable")
        if isinstance(tensor, Module):
            raise TypeError(f"buffer '{name}' cannot be a Module")
        if not isinstance(tensor, ortho.Tensor):
            raise TypeError(
                f"buffer '{name}' expects a Tensor or None, "
                f"got {type(tensor).__name__}")

    # ---- attribute protocol ----

    def __setattr__(self, name, value):
        params = object.__getattribute__(self, '_parameters')
        buffers = object.__getattribute__(self, '_buffers')
        modules = object.__getattribute__(self, '_modules')
        if isinstance(value, Parameter):
            if name in buffers:
                raise KeyError(
                    f"cannot assign a Parameter to '{name}': "
                    "it is already registered as a buffer")
            params[name] = value._data
            modules.pop(name, None)
        elif isinstance(value, Module):
            if name in buffers:
                raise KeyError(
                    f"cannot assign a Module to '{name}': "
                    "it is already registered as a buffer")
            modules[name] = value
            params.pop(name, None)
        elif name in buffers:
            # Reassignment keeps the name a buffer and keeps its persistence.
            self._check_buffer_value(name, value)
            buffers[name] = value
        else:
            params.pop(name, None)
            modules.pop(name, None)
            object.__setattr__(self, name, value)

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

    # ---- enumeration ----

    def parameters(self):
        """Recursively yield all trainable parameter tensors (no duplicates)."""
        seen = set()

        def _collect(module):
            for p in module._parameters.values():
                key = id(p)
                if key not in seen:
                    seen.add(key)
                    yield p
            for m in module._modules.values():
                yield from _collect(m)

        yield from _collect(self)

    def named_parameters(self, prefix=""):
        """Recursively yield (name, tensor) pairs using state_dict-style names."""
        for name, p in self._parameters.items():
            yield prefix + name, p
        for mname, m in self._modules.items():
            yield from m.named_parameters(prefix + mname + '.')

    def named_buffers(self, prefix="", recurse=True):
        """Yield (name, tensor) pairs for registered buffers.

        Includes non-persistent buffers. Skips buffers registered as None.
        Names use the same hierarchical scheme as named_parameters().
        """
        for name, b in self._buffers.items():
            if b is None:
                continue
            yield prefix + name, b
        if recurse:
            for mname, m in self._modules.items():
                yield from m.named_buffers(prefix + mname + '.', recurse=True)

    def buffers(self, recurse=True):
        """Yield registered buffer tensors (no duplicates)."""
        seen = set()
        for _, b in self.named_buffers(recurse=recurse):
            key = id(b)
            if key not in seen:
                seen.add(key)
                yield b

    # ---- training mode ----

    def train(self):
        """Set this module and child modules to training mode. Returns self."""
        self.training = True
        for m in self._modules.values():
            m.train()
        return self

    def eval(self):
        """Set this module and child modules to eval mode. Returns self."""
        self.training = False
        for m in self._modules.values():
            m.eval()
        return self

    def zero_grad(self):
        """Zero out gradients for all parameters."""
        for p in self.parameters():
            if p.grad is not None:
                p.zero_grad()

    # ---- device and dtype ----

    def to(self, device):
        """Move all parameters and buffers to device in-place. Returns self.

        Model-state transformation, not an autograd operation: public to()
        is differentiable, so the moves run under an explicit no_grad region
        and every moved Parameter stays a detached leaf with its requires_grad
        setting preserved.
        """
        with ortho.no_grad():
            for name in list(self._parameters.keys()):
                p = self._parameters[name]
                requires_grad = p.requires_grad
                moved = p.to(device)
                moved.requires_grad_(requires_grad)
                self._parameters[name] = moved
            for name in list(self._buffers.keys()):
                b = self._buffers[name]
                if b is None:
                    continue
                moved = b.to(device)
                if b.requires_grad:
                    moved.requires_grad_(True)
                self._buffers[name] = moved
        for m in self._modules.values():
            m.to(device)
        return self

    def cast(self, dtype):
        """Cast all parameters and floating-point buffers to dtype. Returns self.

        Model-state transformation, not an autograd operation: floating
        cast() is differentiable, so conversions run under an explicit no_grad
        region and every casted Parameter stays a detached leaf with its
        requires_grad setting preserved.
        """
        with ortho.no_grad():
            for name in list(self._parameters.keys()):
                p = self._parameters[name]
                requires_grad = p.requires_grad
                casted = ortho.cast(p, dtype)
                casted.requires_grad_(requires_grad)
                self._parameters[name] = casted
            for name in list(self._buffers.keys()):
                b = self._buffers[name]
                if b is None or b.dtype not in _FLOATING_DTYPES:
                    continue
                casted = ortho.cast(b, dtype)
                if b.requires_grad:
                    casted.requires_grad_(True)
                self._buffers[name] = casted
        for m in self._modules.values():
            m.cast(dtype)
        return self

    def half(self):
        """Cast all parameters to float16. Returns self."""
        return self.cast("float16")

    def bfloat16(self):
        """Cast all parameters to bfloat16. Returns self."""
        return self.cast("bfloat16")

    # ---- persistent state ----

    def _state_entries(self, _prefix=''):
        """Yield (key, owner, kind, local_name, tensor) for each state_dict entry.

        Single source of truth for state_dict() and load_state_dict(), so the
        two can never disagree about which keys exist or what they refer to.
        """
        for name, p in self._parameters.items():
            yield _prefix + name, self, 'parameter', name, p
        for name, b in self._buffers.items():
            if name in self._non_persistent_buffers or b is None:
                continue
            yield _prefix + name, self, 'buffer', name, b
        for mname, m in self._modules.items():
            yield from m._state_entries(_prefix + mname + '.')

    def state_dict(self, _prefix=''):
        """Return {name: tensor} for parameters and persistent buffers.

        The values are LIVE references to the module's current state tensors,
        not copies: mutating a parameter afterwards is visible through a
        state_dict taken earlier, and vice versa. That is deliberate - it is
        what makes ``model.load_state_dict(model.state_dict())`` and in-place
        optimizer updates work.

        The file APIs are different by design: ``save_state_dict`` and
        ``save_checkpoint`` take a SNAPSHOT of the values at save time, so
        mutating the model afterwards cannot change what was written.
        """
        return {key: tensor for key, _, _, _, tensor in self._state_entries(_prefix)}

    def load_state_dict(self, state, strict=True):
        """Copy values from state into matching parameters and buffers."""
        entries = list(self._state_entries())
        current_keys = {key for key, _, _, _, _ in entries}
        if strict:
            missing    = current_keys - set(state.keys())
            unexpected = set(state.keys()) - current_keys
            if missing:
                raise KeyError(f"Missing keys in state_dict: {sorted(missing)}")
            if unexpected:
                raise KeyError(f"Unexpected keys in state_dict: {sorted(unexpected)}")

        # ---- Stage 1: validate and prepare. Nothing is mutated here. ----
        #
        # Every incoming value is turned into an INDEPENDENT tensor already on
        # the destination's device and in the destination's dtype, before any
        # parameter or buffer is touched. Two things fall out of that ordering:
        #
        #  * Self-loading is safe. state_dict() returns the live parameter
        #    objects, so model.load_state_dict(model.state_dict()) has
        #    `src is current`. Preparing a snapshot first means the later
        #    in-place write cannot destroy the values it is reading from - a
        #    float path that did current.mul_(0.0) on an aliased source would
        #    silently zero the whole model.
        #
        #  * A malformed or unconvertible later entry cannot leave earlier
        #    parameters already modified.
        prepared = []
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

            # to() is called UNCONDITIONALLY, including when the device already
            # matches. Tensor::to (src/core/tensor.cpp) routes a same-device
            # move through make_contiguous_copy / cuda_make_contiguous, both of
            # which always allocate a fresh TensorImpl. So this is the single
            # operation that gives independence from the source, materialises a
            # non-contiguous source and reconciles device. Detachment is
            # EXPLICIT: public to()/cast() are differentiable, so the no_grad
            # region - not a side effect - is what keeps a trainable source
            # graph out of the destination.
            with ortho.no_grad():
                restored = src.to(current.device)
                if restored.dtype != current.dtype:
                    restored = ortho.cast(restored, current.dtype)
            prepared.append((owner, kind, local_name, current, restored))

        # ---- Stage 2: commit. ----
        for owner, kind, local_name, current, restored in prepared:
            if kind == 'buffer':
                # Buffers are REPLACED, not copied into: the prepared tensor
                # takes the entry, and the destination's own requires_grad
                # setting is preserved rather than inherited from the source.
                restored.requires_grad_(current.requires_grad)
                owner._buffers[local_name] = restored
                continue

            # Parameters are copied INTO, never replaced. The destination
            # Tensor object stays the parameter, so optimizer references,
            # post-accumulate hooks, requires_grad/is_leaf/grad_fn and .grad
            # all survive the load. _copy_data_ moves the payload only.
            current._copy_data_(restored)
        return self

    def _replace_parameter(self, name, tensor):
        parts = name.split('.')
        module = self
        for part in parts[:-1]:
            module = module._modules[part]
        module._parameters[parts[-1]] = tensor

    # ---- call protocol ----

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
