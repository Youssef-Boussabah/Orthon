import numpy as np
import orthon as ortho


def _fingerprint(parameter):
    grad = parameter.grad
    return None if grad is None else (grad._impl_id, grad._storage_version)


class _Reducer:
    """Stable parameter owners, participation consensus and gradient reduction."""
    def __init__(self, parameters, group, bucket_size_mb=None, overlap=False):
        self.parameters = tuple(parameters)
        self.group = group
        self.overlap = overlap
        self.pending = set()
        self.synced = {id(p): None for _, p in self.parameters}
        self.outstanding = []
        self.launch_count = 0
        self.buckets = None
        if bucket_size_mb is not None:
            limit = int(float(bucket_size_mb) * 1024 * 1024)
            if limit < 1:
                raise ValueError("bucket_size_mb must be positive when provided")
            self.buckets = []
            current, size = [], 0
            for entry in self.parameters:
                p = entry[1]
                nbytes = int(p.numel) * {"float32": 4, "float64": 8,
                                       "float16": 2, "bfloat16": 2}.get(p.dtype, 1)
                if current and size + nbytes > limit:
                    self.buckets.append(current)
                    current, size = [], 0
                current.append(entry)
                size += nbytes
            if current:
                self.buckets.append(current)

    def diagnostics(self):
        return {"buckets": len(self.buckets or ()), "outstanding": len(self.outstanding),
                "async_launches": self.launch_count}

    def mark(self, owner_id):
        self.pending.add(owner_id)

    @staticmethod
    def validate_gradient(grad):
        if grad.dtype not in ("float32", "float64"):
            raise RuntimeError("DDP gradient synchronization only supports float32/float64 gradients; "
                               f"{grad.dtype} gradients cannot be averaged without dtype-preserving "
                               "in-place scaling")

    def reset(self):
        if self.outstanding:
            raise RuntimeError("DDP has pending asynchronous gradient synchronization; "
                               "call wait_for_gradient_sync() before zero_grad()")
        self.pending.clear()
        self.synced = {id(p): None for _, p in self.parameters}
        self.launch_count = 0

    def synchronize(self):
        self.group.ensure_open()
        self.wait()
        if not self.parameters:
            return
        # Consensus precedes data: no hook determines collective order. The
        # second row agrees on packing and the third on profile validity. A
        # rank-local invalid/absent gradient must not split collective order.
        usage = []
        packing = []
        validity = []
        for _, p in self.parameters:
            grad = p.grad
            valid = True
            if grad is not None:
                try:
                    self.validate_gradient(grad)
                    self.group.validate_gradient_dtype(grad.dtype)
                    if (grad.shape != p.shape or grad.device != p.device or grad.dtype != p.dtype
                            or (grad.dtype == "float64" and grad.device.is_cuda())):
                        valid = False
                except (RuntimeError, ValueError, NotImplementedError):
                    valid = False
            validity.append(float(valid))
            usage.append(float(id(p) in self.pending or
                               (grad is not None and _fingerprint(p) != self.synced[id(p)])))
            packing.append(float(p.dtype == "float32" and
                                 (grad is None or (grad.dtype == "float32" and grad.is_contiguous))))
        if self.buckets is not None:
            positions = {id(p): i for i, (_, p) in enumerate(self.parameters)}
            for bucket in self.buckets:
                if any(p.device != bucket[0][1].device for _, p in bucket):
                    for _, p in bucket:
                        packing[positions[id(p)]] = 0.
        with ortho.no_grad():
            control = ortho.from_numpy(np.asarray([usage, packing, validity], dtype=np.float32))
            device = self.parameters[0][1].device
            if device.is_cuda():
                control = control.to(device)
            self.group.all_reduce(control)
            consensus = np.asarray(control.to("cpu").tolist()).reshape(3, -1)
            if np.any(consensus[2] != self.group.world_size):
                raise RuntimeError("DDP gradient synchronization rejected an invalid gradient profile "
                                   "on one or more ranks (dtype, shape, device, or backend support)")
            active = {id(p) for i, (_, p) in enumerate(self.parameters) if consensus[0, i] > 0}
            packable = {id(p) for i, (_, p) in enumerate(self.parameters)
                        if consensus[1, i] == self.group.world_size}
            for _, p in self.parameters:
                if id(p) in active and p.grad is None:
                    grad = ortho.zeros(p.shape, dtype=p.dtype)
                    if p.device.is_cuda():
                        grad = grad.to(p.device)
                    p.set_grad(grad)
            for bucket in self.buckets if self.buckets is not None else [[e] for e in self.parameters]:
                entries = [(p, p.grad) for _, p in bucket if id(p) in active]
                if not entries:
                    continue
                can_pack = (self.buckets is not None and all(id(p) in packable for p, _ in entries)
                            and all(g.device == entries[0][1].device for _, g in entries))
                if can_pack:
                    total = sum(int(g.numel) for _, g in entries)
                    buffer = ortho.empty([total], dtype="float32")
                    if entries[0][1].device.is_cuda():
                        buffer = buffer.to(entries[0][1].device)
                    offset = 0
                    for _, grad in entries:
                        buffer._copy_flat_range_from(offset, grad)
                        offset += int(grad.numel)
                    if self.overlap:
                        work = self.group.all_reduce(buffer, async_op=True)
                        self.outstanding.append((work, buffer, entries))
                        self.launch_count += 1
                    else:
                        self.group.all_reduce(buffer)
                        self._scatter(buffer, entries)
                else:
                    for p, grad in entries:
                        # Raw transports require contiguous physical payloads;
                        # the reducer supports logical strided FP32/F64 inputs.
                        payload = grad if grad.is_contiguous else ortho.contiguous(grad)
                        self.group.all_reduce(payload)
                        payload.mul_(1.0 / self.group.world_size)
                        if payload is not grad:
                            p.set_grad(payload)
            self.wait()
        self.pending.clear()
        self.synced = {id(p): _fingerprint(p) for _, p in self.parameters}

    def _scatter(self, buffer, entries):
        buffer.mul_(1.0 / self.group.world_size)
        offset = 0
        for p, grad in entries:
            if p.grad is None or p.grad._impl_id != grad._impl_id:
                raise RuntimeError("DDP gradient changed before asynchronous bucket synchronization completed")
            buffer._copy_flat_range_to(offset, grad)
            offset += int(grad.numel)

    def wait(self):
        with ortho.no_grad():
            for work, buffer, entries in self.outstanding:
                work.wait()
                self._scatter(buffer, entries)
        self.outstanding.clear()


def _reduce_master_gradients(parameters, group):
    reducer = _Reducer([(str(i), p) for i, p in enumerate(parameters)], group)
    # These are freshly prepared gradients, even if their owner already had a
    # gradient at construction. Absent owners remain lazy on globally unused ranks.
    reducer.pending.update(id(p) for _, p in reducer.parameters if p.grad is not None)
    reducer.synchronize()
