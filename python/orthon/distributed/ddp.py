import orthon as ortho


class DistributedDataParallel:
    """Correctness-first DDP-lite wrapper with optional bucket overlap.

    The wrapper broadcasts parameters through the active process group at
    construction time. By default, parameter post-accumulate hooks synchronize
    gradients after backward. Callers may disable this with sync_grads=False and
    continue to invoke all_reduce_gradients() manually. Bucketed auto-sync can
    optionally launch asynchronous all-reduces during backward; backward still
    waits for outstanding work before returning gradients to callers.
    """

    def __init__(
        self,
        module,
        src=0,
        sync_grads=True,
        bucket_size_mb=None,
        overlap_grad_sync=False,
    ):
        self.module = module
        self._src = src
        self._sync_grads = bool(sync_grads)
        self._bucket_size_mb = bucket_size_mb
        self._overlap_grad_sync = bool(overlap_grad_sync)
        if self._overlap_grad_sync and not self._sync_grads:
            raise ValueError("overlap_grad_sync=True requires sync_grads=True")
        if self._overlap_grad_sync and bucket_size_mb is None:
            raise ValueError("overlap_grad_sync=True requires bucket_size_mb to be provided")
        self._parameters = list(self._ordered_parameters())
        self._parameter_ids = {id(parameter) for _, parameter in self._parameters}
        self._buckets = self._build_buckets(bucket_size_mb)
        self._bucket_parameter_ids = self._build_bucket_parameter_ids()
        self._bucket_index_by_parameter_id = self._build_bucket_index_by_parameter_id()
        self._bucket_ready_parameter_ids = [set() for _ in self._bucket_parameter_ids]
        self._next_bucket_to_launch = 0
        self._outstanding_bucket_work = []
        self._overlap_bucket_launch_count = 0
        self._all_gradients_ready = False
        self._pending_parameter_ids = set()
        self._synced_grad_ids = {}
        self._auto_synced = False
        from . import broadcast

        for _, parameter in self._parameters:
            broadcast(parameter, src=src)
            if self._sync_grads:
                self._register_parameter_hook(parameter)

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

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

    def zero_grad(self):
        if self._has_pending_gradient_sync():
            raise RuntimeError(
                "DDP has pending asynchronous gradient synchronization; "
                "call wait_for_gradient_sync() before zero_grad()"
            )
        self._reset_sync_state()
        if hasattr(self.module, "zero_grad"):
            return self.module.zero_grad()
        for parameter in self.parameters():
            if parameter.grad is not None:
                parameter.zero_grad()
        return None

    def train(self):
        if hasattr(self.module, "train"):
            return self.module.train()
        return self

    def eval(self):
        if hasattr(self.module, "eval"):
            return self.module.eval()
        return self

    def all_reduce_gradients(self):
        """Average existing parameter gradients across the process group.

        Intended usage:
            loss.backward()
            ddp_model.all_reduce_gradients()
            optimizer.step()
            optimizer.zero_grad()
        """
        if self._has_pending_gradient_sync():
            self.wait_for_gradient_sync()
            return None
        if self._auto_synced:
            return None
        self._sync_gradients(skip_already_synced=False)
        return None

    def wait_for_gradient_sync(self):
        """Wait for pending async bucket gradient synchronization to finish."""
        if self._overlap_grad_sync:
            self._launch_ready_buckets()
        self._wait_for_outstanding_bucket_work(mark_auto_synced=True)
        return None

    def _ordered_parameters(self):
        seen = set()
        if hasattr(self.module, "named_parameters"):
            for name, parameter in sorted(self.module.named_parameters(), key=lambda item: item[0]):
                key = id(parameter)
                if key in seen:
                    continue
                seen.add(key)
                yield name, parameter
            return

        for index, parameter in enumerate(self.module.parameters()):
            key = id(parameter)
            if key in seen:
                continue
            seen.add(key)
            yield str(index), parameter

    def _register_parameter_hook(self, parameter):
        if not hasattr(parameter, "_register_post_accumulate_grad_hook"):
            raise RuntimeError(
                "DDP automatic gradient synchronization requires tensor "
                "post-accumulate grad hooks"
            )
        parameter_id = id(parameter)
        parameter._register_post_accumulate_grad_hook(
            lambda parameter=parameter, parameter_id=parameter_id: self._on_parameter_grad_ready(
                parameter_id,
                parameter,
            )
        )

    def _on_parameter_grad_ready(self, parameter_id, parameter):
        if not self._sync_grads:
            return
        if (
            self._overlap_grad_sync
            and self._auto_synced
            and not self._pending_parameter_ids
        ):
            # First hook of a NEW backward pass. The previous pass finished its
            # auto-sync (which is the only way _auto_synced becomes True while
            # _pending_parameter_ids is empty, and implies the outstanding work
            # list was drained), but the per-backward overlap bookkeeping -
            # ready sets and the launch cursor - still describes the finished
            # pass. Without this reset no bucket would ever launch again and
            # ranks would silently train on local gradients. The blocking and
            # bucketed non-overlap paths do not need this: they re-derive
            # readiness from _pending_parameter_ids alone, which the completed
            # pass already cleared.
            self._reset_overlap_pass_state()
        self._pending_parameter_ids.add(parameter_id)
        if self._overlap_grad_sync and self._buckets is not None:
            bucket_index = self._bucket_index_by_parameter_id.get(parameter_id)
            if bucket_index is not None:
                self._bucket_ready_parameter_ids[bucket_index].add(parameter_id)
            self._launch_ready_buckets()
            if len(self._pending_parameter_ids) == len(self._parameters):
                self._all_gradients_ready = True
                self._wait_for_outstanding_bucket_work(mark_auto_synced=True)
            return
        if len(self._pending_parameter_ids) == len(self._parameters):
            self._synced_grad_ids.clear()
            self._sync_gradients(skip_already_synced=False)
            self._pending_parameter_ids.clear()
            self._auto_synced = True

    def _sync_gradients(self, skip_already_synced):
        from . import all_reduce, get_world_size

        scale = 1.0 / float(get_world_size())
        with ortho.no_grad():
            if self._buckets is not None:
                for bucket in self._buckets:
                    if not self._try_sync_bucket(bucket, all_reduce, scale, skip_already_synced):
                        for _, parameter in bucket:
                            self._sync_one_gradient(parameter, all_reduce, scale, skip_already_synced)
                return

            for _, parameter in self._parameters:
                self._sync_one_gradient(parameter, all_reduce, scale, skip_already_synced)

    def _reset_overlap_pass_state(self):
        """Reset the per-backward overlap bookkeeping for a new backward pass.

        Deliberately narrower than _reset_sync_state(): _outstanding_bucket_work
        is left alone (it is provably empty when this is called - _auto_synced
        is only set after the work list drains - and pending async work must
        never be discarded), and _overlap_bucket_launch_count keeps counting
        across passes so callers can observe that every backward launches new
        collectives.
        """
        self._pending_parameter_ids.clear()
        self._synced_grad_ids.clear()
        self._auto_synced = False
        self._bucket_ready_parameter_ids = [set() for _ in self._bucket_parameter_ids]
        self._next_bucket_to_launch = 0
        self._all_gradients_ready = False

    def _reset_sync_state(self):
        self._pending_parameter_ids.clear()
        self._synced_grad_ids.clear()
        self._auto_synced = False
        self._bucket_ready_parameter_ids = [set() for _ in self._bucket_parameter_ids]
        self._next_bucket_to_launch = 0
        self._outstanding_bucket_work.clear()
        self._overlap_bucket_launch_count = 0
        self._all_gradients_ready = False

    def _sync_one_gradient(self, parameter, all_reduce, scale, skip_already_synced):
        if parameter.grad is None:
            return
        self._validate_gradient_sync_dtype(parameter.grad)
        grad_id = parameter.grad._impl_id
        if skip_already_synced and self._synced_grad_ids.get(id(parameter)) == grad_id:
            return
        all_reduce(parameter.grad, op="sum")
        parameter.grad.mul_(scale)
        self._synced_grad_ids[id(parameter)] = parameter.grad._impl_id

    def _build_buckets(self, bucket_size_mb):
        if bucket_size_mb is None:
            return None
        bucket_bytes = int(float(bucket_size_mb) * 1024 * 1024)
        if bucket_bytes < 1:
            raise ValueError("bucket_size_mb must be positive when provided")

        buckets = []
        current = []
        current_bytes = 0
        for name, parameter in self._parameters:
            nbytes = int(parameter.numel) * self._dtype_size(parameter.dtype)
            if current and current_bytes + nbytes > bucket_bytes:
                buckets.append(current)
                current = []
                current_bytes = 0
            current.append((name, parameter))
            current_bytes += nbytes
        if current:
            buckets.append(current)
        return buckets

    def _build_bucket_parameter_ids(self):
        if self._buckets is None:
            return []
        return [
            {id(parameter) for _, parameter in bucket}
            for bucket in self._buckets
        ]

    def _build_bucket_index_by_parameter_id(self):
        index_by_parameter_id = {}
        for bucket_index, parameter_ids in enumerate(self._bucket_parameter_ids):
            for parameter_id in parameter_ids:
                index_by_parameter_id[parameter_id] = bucket_index
        return index_by_parameter_id

    def _try_sync_bucket(self, bucket, all_reduce, scale, skip_already_synced):
        bucket_data = self._collect_bucket_entries(bucket, skip_already_synced)
        if bucket_data is None:
            return False
        entries, total_numel, device, dtype = bucket_data
        if not entries:
            return True

        buffer = self._pack_bucket_entries(entries, total_numel, device, dtype)
        all_reduce(buffer, op="sum")
        buffer.mul_(scale)
        self._scatter_bucket_entries(buffer, entries)
        return True

    def _collect_bucket_entries(self, bucket, skip_already_synced):
        entries = []
        total_numel = 0
        device = None
        dtype = None
        for _, parameter in bucket:
            if parameter.grad is None:
                return None
            grad = parameter.grad
            self._validate_gradient_sync_dtype(grad)
            grad_id = grad._impl_id
            if skip_already_synced and self._synced_grad_ids.get(id(parameter)) == grad_id:
                continue
            if grad.dtype != "float32" or not grad.is_contiguous:
                return None
            if device is None:
                device = grad.device
                dtype = grad.dtype
            elif grad.device != device or grad.dtype != dtype:
                return None
            entries.append((parameter, grad, int(grad.numel), grad_id))
            total_numel += int(grad.numel)
        return entries, total_numel, device, dtype

    def _pack_bucket_entries(self, entries, total_numel, device, dtype):
        buffer = ortho.empty([total_numel], dtype=dtype)
        if device.is_cuda():
            buffer = buffer.to(device)
        offset = 0
        for _, grad, numel, _ in entries:
            buffer._copy_flat_range_from(offset, grad)
            offset += numel
        return buffer

    def _scatter_bucket_entries(self, buffer, entries):
        offset = 0
        for parameter, grad, numel, grad_id in entries:
            if parameter.grad is None or parameter.grad._impl_id != grad_id:
                raise RuntimeError(
                    "DDP gradient changed before asynchronous bucket synchronization "
                    "completed; call wait_for_gradient_sync() before modifying gradients"
                )
            buffer._copy_flat_range_to(offset, grad)
            self._synced_grad_ids[id(parameter)] = parameter.grad._impl_id
            offset += numel

    def _launch_ready_buckets(self):
        if self._buckets is None:
            return
        from . import all_reduce, get_world_size

        scale = 1.0 / float(get_world_size())
        with ortho.no_grad():
            while self._next_bucket_to_launch < len(self._buckets):
                bucket_index = self._next_bucket_to_launch
                ready_ids = self._bucket_ready_parameter_ids[bucket_index]
                if ready_ids != self._bucket_parameter_ids[bucket_index]:
                    return
                bucket = self._buckets[bucket_index]
                bucket_data = self._collect_bucket_entries(
                    bucket,
                    skip_already_synced=False,
                )
                if bucket_data is None:
                    self._wait_for_outstanding_bucket_work(mark_auto_synced=False)
                    for _, parameter in bucket:
                        self._sync_one_gradient(
                            parameter,
                            all_reduce,
                            scale,
                            skip_already_synced=False,
                        )
                else:
                    entries, total_numel, device, dtype = bucket_data
                    if entries:
                        buffer = self._pack_bucket_entries(entries, total_numel, device, dtype)
                        work = all_reduce(buffer, op="sum", async_op=True)
                        self._outstanding_bucket_work.append({
                            "work": work,
                            "buffer": buffer,
                            "entries": entries,
                            "scale": scale,
                        })
                        self._overlap_bucket_launch_count += 1
                self._next_bucket_to_launch += 1

    def _wait_for_outstanding_bucket_work(self, mark_auto_synced):
        if not self._outstanding_bucket_work:
            if mark_auto_synced and self._all_gradients_ready:
                self._pending_parameter_ids.clear()
                self._auto_synced = True
            return
        with ortho.no_grad():
            for item in self._outstanding_bucket_work:
                item["work"].wait()
                buffer = item["buffer"]
                buffer.mul_(item["scale"])
                self._scatter_bucket_entries(buffer, item["entries"])
        self._outstanding_bucket_work.clear()
        if mark_auto_synced and self._all_gradients_ready:
            self._pending_parameter_ids.clear()
            self._auto_synced = True

    def _has_pending_gradient_sync(self):
        return bool(self._outstanding_bucket_work)

    def _dtype_size(self, dtype):
        if dtype in ("float32", "int32"):
            return 4
        if dtype in ("float64", "int64"):
            return 8
        if dtype in ("float16", "bfloat16"):
            return 2
        if dtype in ("uint8", "int8", "bool"):
            return 1
        raise RuntimeError(f"unsupported dtype for DDP bucketing: {dtype}")

    def _validate_gradient_sync_dtype(self, grad):
        if grad.dtype in ("float32", "float64"):
            return
        raise RuntimeError(
            "DDP gradient synchronization only supports float32/float64 gradients; "
            f"{grad.dtype} gradients cannot be averaged without dtype-preserving "
            "in-place scaling"
        )


DDP = DistributedDataParallel


__all__ = [
    "DistributedDataParallel",
    "DDP",
]
