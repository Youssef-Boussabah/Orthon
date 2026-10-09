import weakref

from . import _state
from ._reducer import _Reducer


class DistributedDataParallel:
    """Broadcast trainable parameters and average gradients at backward completion.

    Buckets may enqueue asynchronous reductions after participation consensus;
    every reduction completes before backward returns.
    """
    def __init__(self, module, src=0, sync_grads=True, bucket_size_mb=None,
                 overlap_grad_sync=False):
        self.module = module
        self._sync_grads = bool(sync_grads)
        if overlap_grad_sync and not sync_grads:
            raise ValueError("overlap_grad_sync=True requires sync_grads=True")
        if overlap_grad_sync and bucket_size_mb is None:
            raise ValueError("overlap_grad_sync=True requires bucket_size_mb to be provided")
        if bucket_size_mb is not None and int(float(bucket_size_mb) * 1024 * 1024) < 1:
            raise ValueError("bucket_size_mb must be positive when provided")
        group = _state.get_active()
        entries = list(module.named_parameters()) if hasattr(module, "named_parameters") else [
            (str(i), p) for i, p in enumerate(module.parameters())]
        seen = set()
        owners = []
        for name, p in sorted(entries, key=lambda entry: entry[0]):
            if id(p) not in seen and p.requires_grad:
                seen.add(id(p))
                owners.append((name, p))
        self._reducer = _Reducer(owners, group, bucket_size_mb, bool(overlap_grad_sync))
        reference = weakref.ref(self._reducer)
        def finalize():
            reducer = reference()
            if reducer is not None:
                reducer.synchronize()
        self._finalizer = finalize
        self._finalizer_key = id(self._reducer)
        for _, p in owners:
            group.broadcast(p, src=src)
            def ready(owner_id=id(p), reference=reference):
                reducer = reference()
                if reducer is not None:
                    reducer.mark(owner_id)
            p._register_post_accumulate_grad_hook(ready)
            if self._sync_grads:
                p._register_backward_finalizer(self._finalizer_key, finalize)

    def _attach_outputs(self, output):
        if not self._sync_grads:
            return
        if isinstance(output, (tuple, list)):
            for value in output:
                self._attach_outputs(value)
        elif isinstance(output, dict):
            for value in output.values():
                self._attach_outputs(value)
        elif hasattr(output, "_register_backward_finalizer") and output.requires_grad:
            output._register_backward_finalizer(self._finalizer_key, self._finalizer)

    def __call__(self, *args, **kwargs):
        self._reducer.group.ensure_open()
        output = self.module(*args, **kwargs)
        self._attach_outputs(output)
        return output

    def forward(self, *args, **kwargs):
        return self(*args, **kwargs)

    def parameters(self):
        return self.module.parameters()

    def named_parameters(self, *args, **kwargs):
        if not hasattr(self.module, "named_parameters"):
            raise AttributeError(f"{type(self.module).__name__} has no named_parameters()")
        return self.module.named_parameters(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        if not hasattr(self.module, "state_dict"):
            raise AttributeError(f"{type(self.module).__name__} has no state_dict()")
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        if not hasattr(self.module, "load_state_dict"):
            raise AttributeError(f"{type(self.module).__name__} has no load_state_dict()")
        return self.module.load_state_dict(*args, **kwargs)

    def train(self):
        if hasattr(self.module, "train"):
            return self.module.train()
        return self

    def eval(self):
        if hasattr(self.module, "eval"):
            return self.module.eval()
        return self

    def zero_grad(self):
        self._reducer.reset()
        if hasattr(self.module, "zero_grad"):
            return self.module.zero_grad()
        for parameter in self.parameters():
            parameter.zero_grad()

    def all_reduce_gradients(self):
        self._reducer.synchronize()

    def wait_for_gradient_sync(self):
        self._reducer.group.ensure_open()
        self._reducer.wait()

    def _validate_master_gradient_sync(self):
        if self._sync_grads:
            raise ValueError("MixedPrecisionOptimizer requires DDP(sync_grads=False)")


DDP = DistributedDataParallel
__all__ = ["DistributedDataParallel", "DDP"]
