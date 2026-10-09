"""Real collective workloads; all observations use public APIs and values."""

import json
import multiprocessing as mp
import os
import sys

import numpy as np

import orthon as ortho
import orthon.distributed as dist
from orthon.amp import GradScaler, MixedPrecisionOptimizer
from orthon.distributed.local_backend import LocalBackend, create_store
from orthon.nn import Module, Parameter
from orthon.optim import Adam, SGD


def tensor(values, dtype="float32", device="cpu"):
    array = np.array(values, dtype="float32" if dtype == "bfloat16" else dtype)
    result = ortho.from_numpy(array, dtype="bfloat16") if dtype == "bfloat16" else ortho.from_numpy(array)
    return result if device == "cpu" else result.to(device)


def describe(value):
    if value is None:
        return None
    with ortho.no_grad():
        values = value.to("cpu").tolist()
    return dict(values=values, shape=list(value.shape), dtype=value.dtype,
                device=f"cuda:{value.device.index}" if value.device.is_cuda() else "cpu",
                is_leaf=value.is_leaf, requires_grad=value.requires_grad)


def state_values(model):
    return {name: describe(value) for name, value in model.state_dict().items()}


class Trio(Module):
    def __init__(self, rank=0, dtype="float32", device="cpu", unused=False, reverse=False):
        super().__init__()
        values = {"a": [[1., 2.], [3., 4.]], "b": [1., 2.], "c": np.array(2.)}
        for name in reversed(values) if reverse else values:
            setattr(self, name, Parameter(tensor(np.array(values[name]) + rank * 10, dtype, device)))
        if unused:
            self.unused = Parameter(tensor([7.], dtype, device))
        self.register_buffer("offset", tensor([10. + rank], dtype, device))

    def forward(self, factor, *, active="all"):
        result = self.offset[0]
        if active in ("all", "a"):
            result = result + (self.a * tensor([[1., 2.], [3., 5.]], self.a.dtype,
                                             device_of(self.a)) * factor).sum()
        if active in ("all", "b"):
            result = result + (self.b * tensor([2., -1.], self.b.dtype,
                                             device_of(self.b)) * factor).sum()
        if active == "all":
            result = result + self.c * (.5 * factor)
        return result


def device_of(value):
    return f"cuda:{value.device.index}" if value.device.is_cuda() else "cpu"


class RecordedWork:
    def __init__(self, work, event):
        self.work, self.event = work, event

    def wait(self):
        result = self.work.wait()
        self.event["waited"] = True
        return result

    def is_completed(self):
        return self.work.is_completed()


class ObservedLocal(dist.Backend):
    """A public backend adapter recording real local collective calls."""
    def __init__(self, **kwargs):
        self.delegate = LocalBackend(**kwargs)
        self.events = []
        self.model = None
        global observed
        observed = self

    @property
    def rank(self):
        return self.delegate.rank

    @property
    def world_size(self):
        return self.delegate.world_size

    def all_reduce(self, value, op="sum", async_op=False):
        event = dict(async_op=async_op, waited=False, shape=tuple(value.shape))
        if self.model is not None:
            event["missing_model_gradients"] = sum(p.grad is None for p in self.model.parameters())
        self.events.append(event)
        result = self.delegate.all_reduce(value, op=op, async_op=async_op)
        return RecordedWork(result, event) if async_op else result

    def broadcast(self, value, src=0, async_op=False):
        return self.delegate.broadcast(value, src=src, async_op=async_op)

    def barrier(self):
        return self.delegate.barrier()

    def shutdown(self):
        return self.delegate.shutdown()


observed = None
dist.register_backend("spec_observed_local", ObservedLocal)


class Float32GradientLocal(ObservedLocal):
    """Injected reducer restriction, with real LocalBackend transport."""
    def _validate_gradient_dtype(self, dtype):
        if dtype != "float32":
            raise RuntimeError("injected backend requires float32 gradients")


dist.register_backend("spec_float32_gradient_local", Float32GradientLocal)


def gradients(model):
    return {name: describe(p.grad) for name, p in model.named_parameters()}


def wrapper(model, options):
    return dist.DDP(model, sync_grads=options.get("manual", False) is False,
                    bucket_size_mb=options.get("bucket"), overlap_grad_sync=options.get("overlap", False))


def collective_worker(rank, options):
    dtype = options.get("dtype", "float32")
    device = f"cuda:{rank}" if options.get("nccl") else "cpu"
    shape = tuple(options.get("shape", [2, 2]))
    base = np.arange(1, np.prod(shape, dtype=int) + 1).reshape(shape)
    rounds = []
    for asynchronous in (False, True):
        value = tensor(base * (rank + 1), dtype, device)
        outcome = dist.all_reduce(value, async_op=asynchronous)
        work = None
        if asynchronous:
            initial = outcome.is_completed()
            work = dict(initial=initial, waits=[outcome.wait(), outcome.wait()], done=outcome.is_completed())
        else:
            assert outcome is value
        summed = describe(value)
        broadcasts = []
        for source in range(dist.get_world_size()):
            value = tensor(base + 10 * rank, dtype, device)
            outcome = dist.broadcast(value, src=source, async_op=asynchronous)
            if asynchronous:
                assert outcome.wait() is True and outcome.wait() is True and outcome.is_completed()
            else:
                assert outcome is value
            broadcasts.append(describe(value))
        assert dist.barrier() is None
        rounds.append(dict(summed=summed, broadcasts=broadcasts, work=work))
    return rounds


def bool_broadcast_worker(rank, options):
    values = []
    for source in range(dist.get_world_size()):
        value = tensor([rank == 0, rank != 0], "bool")
        assert dist.broadcast(value, src=source) is value
        values.append(describe(value))
    return values


def facade_worker(rank, options):
    model = Trio(rank, reverse=bool(rank))
    owners = dict(model.named_parameters())
    ddp = dist.DDP(model, src=options.get("src", 0), sync_grads=False)
    initial = state_values(model)
    assert all(dict(ddp.named_parameters())[name] is owner for name, owner in owners.items())
    assert all(ddp.state_dict()[name] is value for name, value in model.state_dict().items())
    assert ddp.module is model
    assert ddp.eval() is model and model.training is False
    assert ddp.train() is model and model.training is True
    expected = model(2).item()
    assert ddp(2).item() == ddp.forward(2).item() == expected
    assert ddp.wait_for_gradient_sync() is None
    ddp(2).backward()
    assert ddp.zero_grad() is None
    assert all(p.grad is None for p in ddp.parameters())
    state = {name: tensor(np.full(value.shape, .5), value.dtype, device_of(value))
             for name, value in ddp.state_dict().items()}
    assert ddp.load_state_dict(state) is model
    assert all(dict(ddp.named_parameters())[name] is owner for name, owner in owners.items())
    return dict(initial=initial, output=expected, loaded=state_values(model),
                parameter_count=len(list(ddp.parameters())))


def training_worker(rank, options):
    device = f"cuda:{rank}" if options.get("nccl") else "cpu"
    model = Trio(rank, options.get("dtype", "float32"), device, unused=options.get("unused", False))
    ddp = wrapper(model, options)
    if observed is not None:
        observed.model = model
    opt = SGD(ddp.parameters(), lr=.125)
    trace = []
    for iteration in range(3):
        if observed is not None:
            observed.events.clear()
        active = options.get("usage", ["all"] * 3)[iteration]
        root = ddp((2 + 4 * rank) * (iteration + 1), active=active)
        root.backward()
        local = gradients(model)
        if options.get("manual"):
            assert ddp.all_reduce_gradients() is None
        reduced = gradients(model)
        for _ in range(2):
            assert ddp.wait_for_gradient_sync() is None
        if not options.get("manual"):
            assert ddp.all_reduce_gradients() is None
        assert gradients(model) == reduced
        before = state_values(model)
        assert opt.step() is None
        trace.append(dict(local=local, reduced=reduced, before=before, after=state_values(model),
                          events=[] if observed is None else list(observed.events)))
        assert opt.zero_grad() is None
        assert all(p.grad is None for p in model.parameters())
    return trace


def accumulation_worker(rank, options):
    model = Trio()
    ddp = wrapper(model, options)
    root = ddp(2 + 4 * rank)
    loss = root * 2 + root * 3
    results = []
    for _ in range(2):
        loss.backward(retain_graph=True)
        if options.get("manual"):
            ddp.all_reduce_gradients()
        results.append(gradients(model))
    root = ddp(2 + 4 * rank)
    (root * 2 + root * 3).backward()
    if options.get("manual"):
        ddp.all_reduce_gradients()
    results.append(gradients(model))
    return results


def tied_worker(rank, options):
    model = Module()
    model.first = Parameter(tensor([1., 2.]))
    model.second = Parameter(model.first)
    ddp = wrapper(model, options)
    observed.model = model
    observed.events.clear()
    ((model.first * 2 + model.second * 3) * (2 + 4 * rank)).sum().backward()
    if options.get("manual"):
        ddp.all_reduce_gradients()
    grad = describe(model.first.grad)
    SGD(ddp.parameters(), lr=.125).step()
    assert model.first is model.second
    return dict(gradient=grad, values=model.first.tolist(), parameters=len(list(ddp.parameters())),
                names=[name for name, _ in ddp.named_parameters()], reductions=sum(event["shape"] == tuple(model.first.grad.shape) for event in observed.events))


def amp_worker(rank, options):
    dtype = options.get("dtype", "float32")
    device = f"cuda:{rank}" if options.get("nccl") else "cpu"
    model = Trio(dtype=dtype, device=device)
    ddp = dist.DDP(model, sync_grads=False)
    opt = MixedPrecisionOptimizer(model, Adam, lr=.125)
    scaler = GradScaler(init_scale=8, growth_factor=4, backoff_factor=.25, growth_interval=2)
    rounds = []
    for iteration in range(4):
        opt.zero_grad()
        if device == "cpu" and dtype == "float16":
            for name, p in model.named_parameters():
                coefficients = {"a": [[1., 2.], [3., 5.]], "b": [2., -1.], "c": np.array(.5)}
                p.set_grad(tensor(np.array(coefficients[name]) * (2 + 4 * rank) * scaler.scale_factor,
                                  dtype, device))
        else:
            scaler.scale(ddp(2 + 4 * rank)).backward()
        if iteration == 1 and rank == 1:
            model.b.set_grad(tensor([float("inf"), 8.], dtype, device))
        before = [describe(p) for p in opt.state_dict()["masters"]]
        assert scaler.step(opt) is None
        state = opt.state_dict()
        rounds.append(dict(before=before, masters=[describe(p) for p in state["masters"]],
                          grads=[describe(p.grad) for p in state["masters"]],
                          moments={name: [describe(p) for p in state["inner"][name]] for name in ("m", "v")},
                          model={name: describe(p) for name, p in model.named_parameters()},
                          steps=state["inner"]["t"], scale=scaler.scale_factor, found_inf=scaler.found_inf))
        assert scaler.update() is None
        rounds[-1]["scaler_after_update"] = scaler.state_dict()
    return rounds


def nccl_validation_worker(rank, options):
    device = f"cuda:{rank}"
    outcomes = {}
    valid = tensor([[1., 2.], [3., 4.]], device=device)
    cases = {"cpu": tensor([1.]), "strided": valid.transpose(0, 1),
             "dtype": tensor([1.], "int32", device),
             "wrong-device": tensor([1.], device=f"cuda:{1-rank}")}
    for name, value in cases.items():
        before = describe(value)
        try:
            dist.all_reduce(value)
        except RuntimeError:
            assert describe(value) == before
            outcomes[name] = "RuntimeError"
        else:
            raise AssertionError(f"NCCL accepted unsupported {name}")
    for operation in (lambda: dist.all_reduce(valid, op="mean"), lambda: dist.broadcast(valid, src=2)):
        try:
            operation()
        except (NotImplementedError, ValueError) as error:
            outcomes[type(error).__name__] = True
        else:
            raise AssertionError("NCCL accepted invalid collective arguments")
    dist.barrier()
    return outcomes


def worker(rank, scenario, options):
    if scenario == "spawn":
        return dict(rank=rank, queried_rank=dist.get_rank(), world=dist.get_world_size(),
                    pid=os.getpid(), argument=options["argument"], initialized=dist.is_initialized())
    if scenario == "worker-error":
        if rank == 1:
            raise ValueError("spec worker failure")
        return "other worker completed"
    if scenario == "local-validation":
        value = tensor([[rank + 1., 2.], [3., 4.]])
        before = describe(value)
        for src in (-1, dist.get_world_size()):
            try:
                dist.broadcast(value, src=src)
            except ValueError:
                assert describe(value) == before
            else:
                raise AssertionError("invalid broadcast source accepted")
        try:
            dist.all_reduce(value, op="mean")
        except NotImplementedError:
            assert describe(value) == before
        else:
            raise AssertionError("unsupported reduction accepted")
        strided = value.transpose(0, 1)
        snapshot = describe(strided)
        try:
            dist.all_reduce(strided)
        except RuntimeError:
            assert describe(strided) == snapshot
        else:
            raise AssertionError("non-contiguous destination accepted")
        assert dist.all_reduce(value) is value
        assert dist.barrier() is None
        return describe(value)
    if scenario == "low-precision-ddp":
        dtype = options.get("dtype", "float16")
        device = f"cuda:{rank}" if options.get("nccl") else "cpu"
        model = Module()
        model.weight = Parameter(tensor([1., 2.], dtype, device))
        ddp = dist.DDP(model, sync_grads=False)
        model.weight.set_grad(tensor([2. + 4. * rank, 1.], dtype, device))
        before, gradient = describe(model.weight), describe(model.weight.grad)
        try:
            ddp.all_reduce_gradients()
        except RuntimeError:
            assert describe(model.weight) == before and describe(model.weight.grad) == gradient
        else:
            raise AssertionError("direct low-precision DDP averaging accepted")
        assert dist.barrier() is None
        return "RuntimeError"
    if scenario.startswith("runtime-"):
        return runtime_worker(rank, scenario, options)
    scenarios = {"collectives": collective_worker, "bool-broadcast": bool_broadcast_worker,
                 "facade": facade_worker, "training": training_worker, "accumulation": accumulation_worker,
                 "tied": tied_worker, "amp": amp_worker, "nccl-validation": nccl_validation_worker}
    return scenarios[scenario](rank, options)


def lifecycle_worker(rank, store, options):
    world = dist.get_world_size()
    states = []
    try:
        dist.init_process_group(backend="local", rank=rank, world_size=world, store=store)
    except RuntimeError:
        assert dist.get_rank() == rank and dist.get_world_size() == world
    else:
        raise AssertionError("double init succeeded")
    assert dist.destroy_process_group() is None
    assert dist.is_initialized() is False
    assert dist.destroy_process_group() is None
    for _ in range(2):
        os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank),
                          LOCAL_WORLD_SIZE=str(world), ORTHON_DISTRIBUTED_BACKEND="local")
        if options.get("explicit"):
            os.environ.update(RANK=str((rank + 1) % world), WORLD_SIZE=str(world + 1))
            backend = dist.init_process_group(backend="local", rank=rank, world_size=world, store=store)
        else:
            backend = dist.init_process_group(store=store)
        assert backend.rank == dist.get_rank() == rank
        assert backend.world_size == dist.get_world_size() == world
        assert dist.barrier() is None
        states.append(dict(rank=dist.get_rank(), world=dist.get_world_size()))
        assert dist.destroy_process_group() is None and dist.is_initialized() is False
        try:
            backend.barrier()
        except RuntimeError:
            pass
        else:
            raise AssertionError("destroyed backend remains usable")
    return states


def runtime_worker(rank, scenario, options):
    """Milestone-owned behavior checked through live, bounded rank processes."""
    if scenario == "runtime-invalid-gradient":
        kind = options["kind"]
        dtype = "float16" if kind == "low-precision" else "float64" if kind == "backend-dtype" else "float32"
        model = Module()
        model.weight = Parameter(tensor([1., 2.], dtype))
        wrapped = dist.DDP(model, sync_grads=False, bucket_size_mb=1.)
        if rank == 0:
            grad = tensor([3.] if kind == "shape" else [3., 4.],
                          "float64" if kind == "dtype-mismatch" else dtype,
                          "cuda:0" if kind == "device" else "cpu")
            model.weight.set_grad(grad)
        before = describe(model.weight.grad)
        observed.events.clear()
        try:
            wrapped.all_reduce_gradients()
        except RuntimeError as error:
            message = str(error)
            assert "invalid gradient profile" in message
        else:
            raise AssertionError("rank-local invalid gradient accepted")
        assert describe(model.weight.grad) == before
        assert [event["shape"] for event in observed.events] == [(3, 1)]
        assert wrapped._reducer.diagnostics()["async_launches"] == 0
        assert wrapped._reducer.diagnostics()["outstanding"] == 0
        value = tensor([rank + 1., 2.])
        dist.all_reduce(value)
        return dict(message=message, value=value.tolist(), control_calls=1, data_calls=0)
    if scenario == "runtime-strided":
        dtype = options["dtype"]
        model = Module()
        model.weight = Parameter(tensor([[1., 2.], [3., 4.]], dtype))
        wrapped = dist.DDP(model, sync_grads=False, bucket_size_mb=1.)
        gradient = tensor([[2. + 4 * rank, 4.], [6., 8.]], dtype)
        if rank == 0:
            gradient = gradient.transpose(0, 1)
        model.weight.set_grad(gradient)
        wrapped.all_reduce_gradients()
        return describe(model.weight.grad)
    if scenario == "runtime-distinct-owners":
        model = Module()
        base = tensor([1., 2.])
        second = base.reshape([2])
        model.a = Parameter(base)
        # A view owns independent AutogradMeta despite sharing physical bytes.
        model.b = Parameter(second)
        assert model.a is not model.b
        wrapped = dist.DDP(model, bucket_size_mb=.000001, overlap_grad_sync=True)
        loss = (model.a * (2 + 4 * rank) + model.b * (10 + 4 * rank)).sum()
        loss.backward()
        assert len(list(wrapped.parameters())) == 2
        return [model.a.grad.tolist(), model.b.grad.tolist()]
    if scenario == "runtime-unused-state":
        model = RuntimeModel()
        wrapped = dist.DDP(model, bucket_size_mb=1, overlap_grad_sync=True)
        opt = Adam(wrapped.parameters(), lr=.1, weight_decay=.2)
        for _ in range(3):
            opt.zero_grad()
            wrapped("a", 2 + 4 * rank)["nested"][0][0].backward()
            opt.step()
        state = opt.state_dict()
        return dict(unused=model.unused.tolist(), gradient=describe(model.unused.grad),
                    moment=describe(state["m"][2]), variance=describe(state["v"][2]),
                    buffer=model.rank_buffer.tolist())
    if scenario == "runtime-mismatch":
        kind = options["kind"]
        value = tensor([rank + 1., 3.], "float64" if kind == "dtype" and rank else "float32")
        if kind == "shape" and rank:
            value = tensor([rank + 1.])
        before = describe(value)
        try:
            if kind == "collective" and rank:
                dist.broadcast(value)
            elif kind == "barrier" and rank:
                dist.barrier()
            elif kind == "source":
                dist.broadcast(value, src=rank)
            else:
                dist.all_reduce(value)
        except RuntimeError as exc:
            assert "mismatch" in str(exc)
            assert describe(value) == before
        else:
            raise AssertionError("rank mismatch accepted")
        try:
            dist.barrier()
        except RuntimeError:
            return "failed-and-unchanged"
        raise AssertionError("failed group accepted another collective")
    if scenario == "runtime-preflight":
        value = tensor([rank + 1., 2.])
        for call in (lambda: dist.all_reduce(value, op="mean"),
                     lambda: dist.broadcast(value, src=dist.get_world_size()),
                     lambda: dist.all_reduce(tensor([[1., 2.], [3., 4.]]).transpose(0, 1)),
                     lambda: dist.all_reduce(tensor([1.], "bfloat16")),
                     lambda: dist.all_reduce(tensor([True], "bool"))):
            try:
                call()
            except (RuntimeError, NotImplementedError, ValueError):
                pass
            else:
                raise AssertionError("invalid collective accepted")
        dist.all_reduce(value)
        return value.tolist()
    if scenario == "runtime-abort":
        if rank == dist.get_world_size() - 1:
            raise ValueError("spec worker failure")
        dist.barrier()
        raise AssertionError("aborted collective returned")
    if scenario == "runtime-amp":
        dtype, mode = options["dtype"], options["mode"]
        model = Module()
        initial_dtype = "float32" if dtype == "bfloat16" else dtype
        model.a = Parameter(tensor([10.], initial_dtype))
        model.b = Parameter(tensor([20.], initial_dtype))
        model.unused = Parameter(tensor([30.], initial_dtype))
        wrapped = dist.DDP(model, sync_grads=False)
        if dtype == "bfloat16":
            model.bfloat16()
        opt = MixedPrecisionOptimizer(wrapped, SGD, lr=.25)
        scaler = GradScaler(enabled=mode != "disabled", init_scale=4)
        records = []
        for iteration in range(3):
            opt.zero_grad()
            scale = scaler.scale_factor if mode != "direct" and mode != "disabled" else 1
            if rank == 0:
                model.a.set_grad(tensor([4. * scale], dtype))
            else:
                model.b.set_grad(tensor([8. * scale], dtype))
            if iteration == 1 and mode == "enabled" and rank == 1:
                model.b.set_grad(tensor([float("inf")], dtype))
            if mode == "direct":
                opt.step()
            else:
                scaler.step(opt)
                scaler.update()
            state = opt.state_dict()
            records.append(dict(model=[describe(p) for p in model.parameters()],
                                masters=[describe(p) for p in state["masters"]],
                                grads=[describe(p.grad) for p in state["masters"]],
                                scale=scaler.scale_factor, found_inf=scaler.found_inf))
        return records
    model = RuntimeModel()
    ddp = dist.DDP(model, sync_grads=not options.get("manual"),
                    bucket_size_mb=options.get("bucket"), overlap_grad_sync=options.get("overlap", False))
    if scenario == "runtime-manual-set":
        model.a.set_grad(tensor([2. + 4 * rank]))
        ddp.all_reduce_gradients()
        first = describe(model.a.grad)
        ddp.all_reduce_gradients()
        ddp.wait_for_gradient_sync()
        ddp.wait_for_gradient_sync()
        assert describe(model.a.grad) == first
        model.a.grad.mul_(2)
        ddp.all_reduce_gradients()
        return [first, describe(model.a.grad), describe(model.unused.grad)]
    records = []
    for iteration in range(4):
        active = ("a" if rank == 0 else "b") if iteration < 2 else "a" if iteration == 2 else "none"
        output = ddp(active, 2. + 4 * rank)
        assert output is model.last_output
        loss = output["nested"][0][0]
        loss.backward(retain_graph=iteration == 0)
        if options.get("manual"):
            ddp.all_reduce_gradients()
        before = [describe(p.grad) for p in model.parameters()]
        ddp.wait_for_gradient_sync()
        ddp.all_reduce_gradients()
        assert [describe(p.grad) for p in model.parameters()] == before
        records.append(before)
    return records


class RuntimeModel(Module):
    def __init__(self):
        super().__init__()
        self.a = Parameter(tensor([1.]))
        self.b = Parameter(tensor([2.]))
        self.unused = Parameter(tensor([3.]))
        self.register_buffer("rank_buffer", tensor([dist.get_rank()]))
        self.input = tensor([1.])
        self.input.requires_grad_(True)

    def forward(self, active, factor):
        loss = self.input.sum()
        if active == "a":
            loss = loss + (self.a * factor).sum()
        if active == "b":
            loss = loss + (self.b * factor).sum()
        self.last_output = {"nested": [(loss,)], "literal": 7}
        return self.last_output


def barrier_worker(rank, arrived):
    world = dist.get_world_size()
    observations = []
    for round_number in range(3):
        value = tensor([rank + round_number])
        dist.all_reduce(value)
        arrived[(round_number, rank)] = True
        assert dist.barrier() is None
        observations.append([arrived.get((round_number, peer), False) for peer in range(world)])
    return observations


def main():
    request = json.loads(sys.stdin.read())
    scenario, options, nprocs = request["scenario"], request["options"], request["nprocs"]
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "ORTHON_DISTRIBUTED_BACKEND"):
        os.environ.pop(name, None)
    assert not dist.is_initialized()
    if scenario in ("lifecycle", "barrier"):
        manager, store = create_store()
        try:
            if scenario == "lifecycle":
                result = dist.spawn(lifecycle_worker, args=(store, options), nprocs=nprocs)
            else:
                result = dist.spawn(barrier_worker, args=(manager.dict(),), nprocs=nprocs)
        finally:
            manager.shutdown()
    else:
        backend = ("nccl" if options.get("nccl") else "spec_float32_gradient_local"
                   if options.get("float32_gradient_backend") else "spec_observed_local"
                   if options.get("observe") else "local")
        try:
            result = dist.spawn(worker, args=(scenario, options), nprocs=nprocs, backend=backend)
        except RuntimeError as error:
            if scenario not in ("worker-error", "runtime-abort"):
                raise
            assert "ValueError" in str(error) and "spec worker failure" in str(error)
            result = {"error": "RuntimeError", "worker_exception": "ValueError"}
    assert not dist.is_initialized() and not mp.active_children()
    print("ORTHON_SPEC_RESULT=" + json.dumps(result))


if __name__ == "__main__":
    main()
