# CUDA runtime

CUDA is an optional backend. `cuda_enabled()` reports compile support, not device
availability. `nccl_enabled()` additionally reports NCCL compilation.
See [distributed](distributed.md) for NCCL dependencies and hardware requirements.

## Context and queue ownership

`Runtime` caches a CudaContext per exact CUDA ordinal. Each context owns a canonical
compute CudaStream, allocator, ScratchArena, capabilities and cuBLAS access.
Ordinary CUDA kernels, copies, memsets and cuBLAS calls enqueue on this explicit
compute queue. `CudaContext::create_stream()` creates an auxiliary queue sharing
the lower-level resource owner. Stream copies share a permanently device-bound
RAII lifetime; surviving streams/events can retain resources without retaining
a whole Runtime or context.

`DeviceGuard` selects and restores the physical CUDA ordinal around native calls.
Correct device selection is separate from queue ordering. Helpers propagate exact
Device objects; the string `"cuda"` resolves to `cuda:0`.

One cuBLAS handle per context is configured under a CublasLease that retains the
resource owner, device guard and mutex across configuration/enqueue. The handle
is bound to the canonical compute stream. This avoids concurrent handle mutation.
See `include/orthon/runtime/cuda_context.h` and `src/runtime/cuda_context.cpp`.

## Completion and memory reuse

A CompletionToken owns immutable event state after recording. It can be queried,
waited on by the host or used to enqueue an ordering wait on another stream.
An empty token is immediately complete. Events are not rerecorded underneath
observers. CudaTimingEvent is a separate private diagnostic timing owner; mutable
timing records are not memory-retirement tokens.

MemoryPool caches checked 64-byte size classes in exact-device bins. Live CUDA
allocations record their queue; release records completion of its consumers.
Reuse on the same queue needs no host wait, because queue order suffices. Cross-
queue reuse enqueues an event dependency within the shared resource domain;
different resource owners use a host completion wait. Eviction, cache clearing and raw
reclamation wait for outstanding completion when required. CPU completion is
immediate. See `src/memory/pool.cpp` and `include/orthon/memory/allocation.h`.
There is no block splitting/coalescing or claim of allocator parity with larger
frameworks.

ScratchArena provides exclusive ScratchLease regions for temporary kernel buffers.
A lease keeps the arena/allocation alive, records completion on release and orders
later reuse. Destruction/reclamation respects the last consumer. Reduction partials
use this service. The legacy `legacy_allocate`/`legacy_reset` diagnostic cursor
is separate from the lease path; it is not an ordinary operator workspace owner.
See `src/memory/scratch.cpp`.

## Transfers and host visibility

CudaContext centralizes transfer policy:

| Transfer | Completion boundary |
| --- | --- |
| CPU to CUDA | Snapshot semantics: enqueue then wait on compute, so the host source can be reused after return |
| CUDA to CPU | Blocking compute-queue copy before host values are read |
| Same-device CUDA copy / pitched copy / memset | Asynchronous compute-queue work |
| Cross-device peer copy | Wait for source compute, enqueue on the exact destination, wait before releasing source storage |

`to()` takes an independent logical copy. `item()`, `tolist()` and checkpoint
snapshots materialize host values through these boundaries. Public
`cuda_synchronize()` remains the compatibility device synchronization operation:
current physical device, CPU build no-op, legacy ignored CUDA errors. It is not
an API for selecting streams. There is no public Python stream-management API.

## Dispatch and the contiguity contract

Semantic operations validate and normalize operands before dispatch. CPU kernels
can walk logical strides where supported. CUDA arithmetic kernels consume packed
contiguous operands via semantic normalization; autograd keeps the original input
identity and saved Buffer watches. Contiguous CUDA copies of storage-only dtypes
are supported; strided packing is restricted to float32/float16/bfloat16.

In-place CUDA destinations must already be contiguous. Partial CUDA `__setitem__`
is unsupported. Full scalar float32 access uses checked logical offsets and a
host-visible transfer boundary. Support varies by operation and dtype; consult
the [coverage matrix](cuda_coverage.md), not the allocation surface.

## NCCL ordering

A native NCCL ProcessGroup owns one communicator and one communication stream.
Compute completion orders communication before the collective; communication
completion orders subsequent compute. Work retains completion/storage, without
creating a stream for each handle. Dropping Work does not perform a host wait;
group shutdown drains and retires resources. This ordering is implemented and
covered by injected native transport tests. Physical NCCL validation requires an NCCL-enabled CUDA build and multiple
suitable GPUs. [Distributed](distributed.md) explains public Work,
Reducer participation and hardware requirements.

## Observation

Private `orthon._diagnostics` exposes selected pool/planner/kernel observations for
tests and benchmarks. Nsight owns external timelines and kernel counters;
[profiling](nsight_profiling.md) separates instrumented hot loops from ordinary
[benchmark latency](benchmarking.md). Stream architecture alone does not establish
speedups or multi-GPU execution evidence.
