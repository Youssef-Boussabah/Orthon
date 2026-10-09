# Changelog

## 1.0.0rc1

First public release candidate after 0.1.0, intended for developer evaluation.
It is not a stable or production-qualified release. PyPI publication and stable
1.0.0 support are outside this RC1 scope.

This release redesigns runtime ownership and execution around explicit services.
CPU/CUDA tensors, views, first-order autograd, NN modules, SGD/Adam, explicit AMP,
pickle-free checkpoints and single-node distributed functionality already existed
in 0.1.0. The changes below concern their implementation and contracts rather
than announcing those capabilities for the first time.

### Tensor ownership and layout

- Replace the canonical Storage-based TensorImpl representation with shared
  Buffer ownership, immutable checked Layout, logical TensorMeta and separate
  AutogradMeta. Storage remains a native compatibility facade.
- Keep storage-sharing views while separating physical allocation/mutation
  generation from each tensor's gradient and producer identity. Validate shape,
  stride, offset and accessed spans with checked arithmetic, including empty views.

### Operator dispatch and execution planning

- Separate semantic validation and graph attachment from CPU arithmetic and CUDA
  launches. Backend implementations now live under src/backends; autograd formulas
  call canonical semantic operations rather than owning raw backend execution.
- Add operator schemas, typed signatures and a frozen KernelRegistry. Dispatch
  resolves explicit device, dtype, layout, transfer target and capability profiles;
  unsupported profiles are distinguished from corrupt or missing registrations.
- Extract MatmulProblem/MatmulPlan/MatmulPlanner and shared ReductionPlan metadata
  from operator execution. CPU reference, custom CUDA and cuBLAS routes remain
  bounded by supported shapes/dtypes; arbitrary batch-broadcast matmul is unsupported.
- Define a shared extrema selector for max/argmax: first logical NaN takes
  precedence, otherwise the first maximum wins, including all-negative-infinity
  inputs. Forward winners and backward routing use the same policy.

### Allocation and CUDA execution lifetimes

- Introduce move-only Allocation objects that retain their allocator, with CPU
  and exact-device CUDA allocators served by explicit Runtime contexts.
- Move pool and workspace ownership into runtime services. ScratchLease gives
  each operation an exclusive region instead of relying on a global scratch cursor.
- Add internal CUDA streams and immutable CompletionTokens for queued consumers,
  scratch reuse and allocation retirement. CudaContext owns compute resources,
  capability state and serialized cuBLAS access; no public Python stream API is added.

### Autograd and training state

- Move recursive traversal embedded in Tensor::backward into an AutogradEngine
  with per-call GraphTask discovery, captured edges, dependency scheduling and a
  ready queue. Task-owned tensor handles protect identity during execution.
- Preserve no-grad backward, saved-value mutation checking, gradient accumulation
  and retained graphs; strengthen captured-history validation, independent leaf
  gradient publication and private completion finalizers. Higher-order autograd
  remains unsupported.
- Keep registered parameter Python owners stable through supported Module device
  and dtype transforms. Prepare all transformed representations before publication,
  preserving ties and avoiding stale optimizer bindings after owner replacement.
- Introduce explicit optimizer bindings, paired Adam moment state and prepared
  state-load protocols. State snapshots do not alias live optimizer moments.
  SGD and Adam are preserved algorithms; SGD momentum is not introduced.
- Reconcile explicit AMP model/master ownership through prepared state services.
  FP32 masters and gradient scaling are preserved; autocast is still unsupported.

### Checkpoint integrity and restore

- Split the monolithic checkpoint implementation into SerializationService,
  ArchiveCodec, ValueCodec, TensorSnapshotCodec and RestoreTransaction services.
  ZIP/JSON/raw tensor payloads, checksums and pickle rejection already existed.
- Validate every present known archive section before changing recipients,
  including sections not requested for restore. Prepare model and optimizer
  state together, then commit with reverse rollback attempts on synchronous errors.
- Preserve the ml-runtime-checkpoint wire identifier and version 1, the four file
  APIs and bit-preserving tensor snapshots. Stricter validation can reject corrupt
  archives previously accepted; this is not a blanket compatibility guarantee for
  custom callbacks or a sharded/distributed checkpoint service.

### Distributed ownership

- Separate public transports from captured ProcessGroup lifecycle and failure
  authority. DDP captures its group rather than following later default replacement.
- Move parameter participation, validity consensus, bucket scheduling, averaging
  and outstanding Work into Reducer. Local DDP remains host transport.
- Native NCCL ProcessGroup owns the communicator and communication queue; Work
  retains completion/storage lifetime rather than owning a stream per operation.
  Compute/communication ordering uses CompletionTokens.
- Bucketed asynchronous launches occur at backward finalization and finish before
  backward returns; the overlap option does not promise general overlap with
  preceding backward compute. Current physical NCCL/multi-GPU execution is unvalidated.

### Safety, tests and packaging

- Share checked pooling geometry across forward/backward and CPU/CUDA routes:
  reject overflowing padded extents, invalid output geometry and CUDA narrowing
  before execution. Harden convolution geometry and typed fallback admission.
- Validate full-width CUDA embedding/NLL indices before data access, and complete
  convolution block-reduction readers before shared workspace is reused.
- Expand executable specification and architecture coverage for ownership,
  aliasing, graph lifetime, optimizer state, archive integrity and distributed
  ordering. Keep regression/parity tests, native tests and installed-package smoke
  as distinct layers with explicit loaded-package and hardware/profile gates.
- Repair plain CMake Python staging, align Python version 1.0.0rc1 with CMake
  numeric base 1.0.0 and its separate full prerelease value, and package the complete
  pybind11 redistribution notice alongside the unchanged Orthon MIT license.

### Qualification boundaries

Strongest inherited qualification is Linux x86_64, Python 3.13.15, NumPy 2.5.3,
GNU 16.2.1 and CPU, with scoped AVX2, NumPy 2.5.1 and Clang 22.1.8 checks.
Current-source Windows/MSVC and the declared minimum dependency versions remain
unqualified. CUDA evidence is limited to one NVIDIA SM89 GPU with toolkit 13.3.73.

Inherited full CPU results: 6,134 passed / 3,684 skipped. Inherited full CUDA
results: 10,024 passed / **98 failed** / 78 skipped, including 76 unsupported cast
combinations and 22 fill cases. These campaigns were not rerun for release
presentation work, and the full CUDA suite is not all-green.

Python shutdown U5 remains OPEN_UNRESOLVED_BOUNDED, including sanitizer exit 23;
leak freedom is not established. Approximately 10-microsecond admission overhead
was observed on selected small SM89 embedding/NLL workloads; this is hardware
and workload specific. Operator/dtype exclusions, physical NCCL qualification
and other support limits remain in [support](SUPPORT.md) and the
[coverage matrix](docs/cuda_coverage.md). See [release notes](RELEASE_NOTES.md)
for the shorter release overview.

## 0.1.0 — 2026-09-02

Initial published Orthon release. Its historical validation does not qualify the
current candidate; new release claims follow current-source evidence.
