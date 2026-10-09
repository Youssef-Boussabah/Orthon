# Architecture

This document describes the Orthon 1.0.0rc1 release tree. Source and
executable tests define behavior. The [documentation index](README.md) links
API and subsystem guides.

## Tensor ownership

`Tensor` is a value handle to a `TensorImpl`. The implementation holds four
canonical responsibilities:

| Owner | Responsibility | Source |
| --- | --- | --- |
| `Buffer` (shared) | Physical Allocation, exact Device, requested/capacity bytes, atomic mutation generation | `include/orthon/core/buffer.h` |
| `Layout` | Immutable shape, strides, element offset, numel and checked accessed span | `include/orthon/core/layout.h` |
| `TensorMeta` | Logical dtype | `include/orthon/core/tensor_meta.h` |
| `AutogradMeta` | Per-identity requires-grad/leaf state, gradient, producer history, hooks and finalizers | `include/orthon/core/autograd_meta.h` |

`include/orthon/core/tensor_impl.h` composes these owners. Distinct tensor owners
can share one Buffer; sharing physical bytes does not merge their gradients or
history. Transpose, slice, indexing and legal reshape views construct new Layout
and tensor metadata while retaining the Buffer. Layout validates bounds using
capacity and element size, with checked arithmetic and explicit empty ranges.
`contiguous()` materializes a strided value when needed; a contiguous value may
retain its identity. `to()` makes an independent logical copy, even on the same
device. Same-dtype `cast()` normalizes contiguity.

Successful logical writes bump Buffer generation, including same-value and empty
writes; metadata-only operations do not. Autograd selectively watches the Buffer
of values a backward formula reads. `Storage` in `include/orthon/core/storage.h`
is a native compatibility facade, not the canonical tensor owner.

## Allocation and runtime

An `Allocation` is a move-only checked-out lifetime retaining its Allocator.
Allocators delegate caching to `MemoryPool` or raw allocation as appropriate;
Buffer does not maintain a second allocation ledger. The pool uses checked
64-byte size classes, exact-device bins, bounded cached bytes and allocation/reuse
statistics. It does not split or coalesce blocks.

`Runtime` caches one `CpuContext` and a `CudaContext` per CUDA ordinal. Each context
provides allocator and ScratchArena services; a CUDA context also owns its
canonical compute CudaStream, capability information and serialized cuBLAS access.
See `include/orthon/runtime/runtime.h` and [CUDA runtime](cuda_runtime.md) for
CompletionToken ordering, completion-aware retirement and transfer boundaries.
ScratchLease owns an exclusive region through its queued consumers; resetting a
legacy diagnostic scratch cursor is not how ordinary kernels release workspace.

## Semantic operations and execution

The execution route is:

```text
public operation -> semantic validation/graph attachment -> DispatchRequest
                 -> Dispatcher -> frozen KernelRegistry -> selected backend kernel
```

`src/ops/` owns semantic orchestration. Arithmetic loops and CUDA launches live
in `src/backends/cpu/` and `src/backends/cuda/`; bindings convert Python arguments
and call the semantic operations. Composite operations use canonical primitives
rather than gaining an unrelated execution owner.

`src/dispatch/builtin_registration.cpp` installs schemas and backend registrations.
`KernelRegistry` checks schema identity, signature, expected coverage and ambiguous
constraints before freezing. Dispatcher selects using backend, dtype, layout,
target dtype/device and capabilities. A declared unsupported profile raises
`UnsupportedOperation`; corrupt or missing expected registrations are internal
errors. The selected kernel still performs its native contract checks.

CUDA operand packing uses logical contiguous values while autograd retains the
user's original tensor identity. Packing an input is different from replacing an
in-place destination: strided CUDA destinations reject before losing a write.

## Matmul and reduction planning

`MatmulProblem` records rank, batch, m/k/n, dtype, exact device and operand layouts.
`MatmulPlannerPolicy` and `MatmulPlanner` select a `MatmulPlan` with CPU reference,
CUDA custom tiled or cuBLAS algorithms, packing facts, work estimates and dtype/
capability/threshold guards. Plans contain metadata, not Tensor or stream owners.
Execution checks the plan against the operands. The supported forms are 2-D and
3-D with equal batch size; arbitrary batch broadcasting is unsupported.
See `include/orthon/planning/matmul_planner.h`.

`ReductionPlan` normalizes an optional single dimension, negative dimensions,
keepdim, output/broadcast shapes and outer/reduce/inner geometry. It is an
immutable metadata value, separate from graph state and ScratchLease execution
workspace. Sum/mean/max/argmax share this normalization. Extrema select the first
logical maximum, with the first logical NaN taking precedence; argmax returns
int64 indices without a differentiable output. See
`include/orthon/reduction/reduction_plan.h` and `src/reduction/reduction.cpp`.

## Autograd and training owners

`Tensor::backward` delegates to AutogradEngine. Each call discovers and evaluates
one GraphTask using captured direct edges, dependencies and a ready queue. It
preflights saved Buffer versions and leaf accumulation support, evaluates with
grad mode disabled, commits independent leaf gradients, runs leaf hooks, runs
unique private task finalizers and then releases history unless retained.
[Autograd](autograd.md) specifies the failure and accumulation boundaries.

Python Module registration owns stable parameter Tensor objects. SGD/Adam retain
bindings to those objects; AMP associates each model owner with a current dtype,
exact device and FP32 master. Supported Module transforms change representations
without replacing parameter Python owners. [Training](training_stack.md) explains
persistent buffers, ties and prepared state loading.

## Checkpointing

`python/orthon/checkpoint.py` delegates to private SerializationService,
ArchiveCodec, ValueCodec, TensorSnapshotCodec and RestoreTransaction services in
`python/orthon/_serialization/`. Loading decodes the whole present known archive,
validates it, prepares all requested owners and then commits, attempting reverse
rollback on synchronous commit errors. [Checkpoint format](checkpoint_format.md)
defines the immutable format, byte fidelity and local-file limits.

## Distributed ownership

The Python facade captures a private ProcessGroup around a public Backend
transport. DDP captures that group and delegates parameter participation,
consensus, buckets, averaging and outstanding Work to a private Reducer.
Native NCCL ProcessGroup owns a communicator and one communication CudaStream;
Work owns immutable completion and storage lifetime, not a stream. Compute and
communication exchange CompletionTokens. [Distributed](distributed.md) covers
exact ordering, overlap limits and current hardware/dependency evidence.

## Private observability and boundaries

`orthon._diagnostics` provides benchmark/test observations and forced routes.
It is private, as are the native extension bridge, prepared-state protocol,
ProcessGroup and Reducer implementations. Benchmark methodology lives in
[benchmarking](benchmarking.md); instrumentation is not ordinary latency evidence.

There is no higher-order autograd, autocast, public Python stream management,
multi-node rendezvous or sharded checkpoint service. Operator/dtype support is
bounded by the [coverage matrix](cuda_coverage.md), not by PyTorch API similarity.
