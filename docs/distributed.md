# Distributed runtime

Orthon exposes single-node collectives and a DDP facade. Current source is in
`python/orthon/distributed/` and `src/distributed/`; source and executable tests
define the contracts below. There is no multi-node rendezvous or sharded checkpoint
service. [API reference](api.md) lists the public names and signatures.

## Public facade, ProcessGroup and transport

`init_process_group(backend=None, **kwargs)` resolves a registered backend name,
constructs a private captured ProcessGroup and
sets the active default group. Environment defaults come from
`rank_config_from_env`: RANK, WORLD_SIZE, LOCAL_RANK and LOCAL_WORLD_SIZE.
`ORTHON_DISTRIBUTED_BACKEND` supplies the backend name when it is omitted. With
neither an explicit name nor that environment setting, initialization rejects.
The return value is the Backend instance, while lifetime/failure authority belongs
to the private ProcessGroup wrapper.

Public `all_reduce`, `broadcast`, `barrier`, rank queries and destruction delegate
to the active group. DDP captures the initialized group at construction. Replacing
the global default later does not retarget an existing DDP reducer. Shut-down or
failed groups reject subsequent use.

Custom transports implement the public `Backend` contract (rank/world_size,
all_reduce, broadcast, barrier and shutdown), and register a factory through
`register_backend(name, factory)`. Applications do not construct `_ProcessGroup`
or `_Reducer`; both are private service owners.

## Work and NCCL queue ordering

Synchronous all_reduce/broadcast return the modified tensor. With `async_op=True`,
they return public Work, with `wait()` and `is_completed()`. Local transport
completes synchronously and returns an already-completed handle. Native NCCL Work
retains immutable completion, storage and group failure/lifetime state. Dropping
it does not perform a host wait or destroy a separately owned communication queue.

Native `NcclProcessGroup` owns one communicator and one communication CudaStream,
plus its compute queue handle and barrier storage. Every collective follows:

```text
compute CompletionToken -> communication wait -> NCCL enqueue
                        -> communication CompletionToken -> compute wait
```

Later compute is ordered after the collective without a normal device-wide wait.
Shutdown drains communication, retires the communicator, settles surviving Work
and releases group resources. Status/transport failures remain sticky; exceptional
cleanup is distinct from the normal enqueue path. Bindings are thin native adapters.
See `src/distributed/nccl_process_group.cpp` and [CUDA runtime](cuda_runtime.md).

NCCL transport accepts contiguous CUDA F32/F16/BF16 tensors on the communicator's
exact device; all_reduce supports sum only. Barrier uses a small dummy reduction.
The transport profile is wider than the DDP gradient averaging profile: NCCL DDP
requires F32 gradients.

## Reducer, participation and buckets

`DistributedDataParallel(module, src=0, sync_grads=True, bucket_size_mb=None,
overlap_grad_sync=False)` captures sorted, unique trainable parameter Python
owners and broadcasts their initial values. DDP registers private leaf
post-accumulate hooks to mark participation and private task finalizers to invoke
the Reducer after backward leaf commits. DDP output attachment ensures finalization
also occurs when an output reaches none of its parameters.

The private Reducer owns pending participation, synchronized gradient fingerprints,
bucket planning, outstanding Work, averaging and scatter. Before gradient data
collectives it exchanges deterministic usage, packing and validity consensus.
Invalid gradient profiles reject consistently before data collective order splits.
Raw collectives still require callers to agree on order/shape/type; this is not
a guarantee against arbitrary distributed faults or divergent programs.

A globally unused owner schedules no gradient data collective and receives no
synthetic grad. If used on any rank, all ranks participate. A rank with no local
grad creates a zero contribution; an existing accumulated grad remains part of
the accumulator and is not discarded merely because this pass did not reach it.
Use zero_grad for independent training steps. Fingerprints prevent unchanged
already-synchronized gradients from being needlessly treated as fresh local work.

DDP supports F32/F64 logical gradients where the transport/device supports them;
CUDA F64 accumulation and NCCL F64 averaging are unsupported. Eligible buckets
pack F32 contiguous gradients on compatible devices. Other eligible profiles use
individual contiguous payloads, materializing strided logical grads as needed.
Reduced gradients are divided by world_size and scattered to their owners.

`overlap_grad_sync=True` requires sync_grads and a bucket size. Asynchronous bucket
launches begin at backward finalization after participation consensus, in
deterministic bucket order. Wait and scatter finish before backward returns.
The name does not imply general overlap of collectives with the entire preceding
backward compute graph. `all_reduce_gradients()` exposes explicit synchronization;
`wait_for_gradient_sync()` checks lifetime and drains outstanding work.

For AMP, construct DDP with `sync_grads=False`, then MixedPrecisionOptimizer and
GradScaler. FP32 master gradients are reduced after conversion/unscale. Enabled
scaling agrees on nonfinite decisions; disabled scaling and BF16 still synchronize
masters. Direct low-precision DDP gradient averaging is unsupported.

## LocalBackend smoke

LocalBackend is a correctness-first host/multiprocessing transport. It uses a
shared condition/store, descriptor handshakes and ordered sequence generations.
Mismatched descriptors reject the group. Collective waits and worker-failure
supervision are bounded; peer failures trigger shared abort and cleanup. CUDA
payloads stage through host memory, so local throughput is not NCCL performance.

Save this complete example as `local_demo.py`, then run `python local_demo.py`
with an installed Orthon package:

```python
import orthon as o
import orthon.distributed as d


def worker(rank):
    value = o.full([1], float(rank + 1))
    work = d.all_reduce(value, async_op=True)
    assert work.wait()
    assert work.is_completed()
    d.barrier()
    return value.item()


if __name__ == "__main__":
    print(d.spawn(worker, nprocs=2, backend="local"))
```

`spawn` initializes each worker's group and destroys it at exit. It returns rank-
ordered results and terminates surviving peers on worker failure. It cannot run
inside an already-initialized parent group. The existing application demo is
`examples/ddp_local_demo.py`.

## NCCL build and launch boundary

When toolkit, NCCL headers/library and multiple suitable physical GPUs are present:

```bash
cmake --preset cuda-nccl-release
cmake --build --preset cuda-nccl-release
```

Use an environment-configured training script with the single-node launcher.
A repository measurement driver with that configuration is:

```bash
python -m orthon.distributed.run --nproc_per_node 2 --backend nccl benchmarks/benchmark_distributed_ddp.py --mode bucketed --bucket-size-mb 25
```

The launcher sets rank environment variables and a local-file NCCL unique-ID
rendezvous. The NCCL benchmark is a measurement tool, not the minimal application
quick start. See [benchmarking](benchmarking.md).

## Validation requirements

LocalBackend tests use CPU multiprocessing. Injected native transports exercise
ProcessGroup ownership, collective order and failure handling. Physical NCCL
validation requires a CUDA+NCCL build, NCCL headers/library and multiple usable
GPUs on one node; injected transports do not establish NCCL runtime correctness.
Tests skip unavailable hardware explicitly. See [testing](testing.md).
