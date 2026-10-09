# Orthon 1.0.0rc1 release notes

Orthon 1.0.0rc1 is the first public release candidate after the published 0.1.0
baseline, intended for developer evaluation. It is not a stable or
production-qualified release. PyPI publication and stable 1.0.0 support are
outside this RC1 scope.

## Release scope

Orthon provides a C++17 tensor runtime and Python API with CPU and optional CUDA
backends, storage-sharing views, first-order autograd, neural-network modules,
SGD/Adam, explicit mixed precision, local multiprocessing collectives, optional
single-node NCCL and pickle-free checkpointing. See the
[audited coverage matrix](docs/cuda_coverage.md) for exact operator/dtype limits.

## What Changed Since v0.1.0

Version 0.1.0 already provided CPU/CUDA tensors, autograd, neural-network modules,
SGD/Adam, explicit mixed precision, pickle-free checkpoints and local/NCCL
collectives. RC1 is a major redesign of the ownership and execution underneath
those features, with stronger contracts and targeted correctness fixes.

- **Tensor identity is explicit.** Shared Buffer owns physical bytes and mutation
  generation; Layout owns checked shape/stride/offset geometry. Tensor and autograd
  metadata remain independent for each owner, even when views share storage.
- **Operations have one execution boundary.** Semantic operations validate inputs
  and attach history, then resolve typed kernels through schemas and a frozen
  registry. CPU/CUDA kernels and matmul/reduction planning have separate owners.
- **Allocation and queued work have defined lifetimes.** Allocations retain their
  allocator; runtime contexts own device resources and leased scratch workspace.
  Internal CUDA completion tokens order workspace reuse and compute/communication.
  This does not establish leak freedom or add public stream control.
- **Backward uses a dedicated engine.** Per-call graph tasks capture producer
  edges, schedule dependencies and commit independent leaf gradients, retaining
  saved-value mutation checks and no-grad backward. Higher-order autograd remains
  unsupported.
- **Training owners survive supported transforms.** Module parameters keep their
  Python identity through device/dtype changes, so existing optimizer bindings
  remain attached. Optimizer moments, AMP masters and state loading use explicit
  ownership and preparation protocols.
- **Checkpoint restore validates before publishing state.** Separate archive,
  value and tensor services validate all present known sections; model and optimizer
  prepare together, with rollback attempts on synchronous commit failures.
  The existing pickle-free ml-runtime-checkpoint version 1 format and file APIs
  are preserved. Invalid archives can face stricter rejection; custom callbacks
  and distributed/sharded checkpoints have no universal recovery guarantee.
- **Distributed execution has dedicated owners.** Captured ProcessGroup, Reducer
  and Work services define lifecycle, participation and completion. Local DDP
  remains host transport. Asynchronous buckets finish before backward returns;
  the overlap option is not a general compute-overlap promise. Physical NCCL
  qualification on the current source remains incomplete.
- **Safety fixes target concrete edge cases.** Checked pooling/convolution
  geometry rejects overflow before execution. CUDA embedding/NLL admission checks
  full-width indices, convolution reductions synchronize shared workspace reuse,
  and max/argmax share a first-NaN/first-maximum winner policy. These scoped fixes
  do not imply that every CUDA operation or dtype is supported.

The familiar supported tensor/training API and checkpoint wire identity remain;
this is not a claim of complete behavioral compatibility, private-API stability
or framework parity. Detailed changes and their boundaries are in the
[changelog](CHANGELOG.md), [architecture guide](docs/architecture.md) and
[operator/dtype matrix](docs/cuda_coverage.md).

## Installation and qualification

Build the source with CPU features selected explicitly:

```bash
python -m pip install . -C cmake.define.WITH_CUDA=OFF -C cmake.define.WITH_NCCL=OFF
```

The CPU distribution artifacts are an sdist and a Linux wheel. The
`cp313-cp313-linux_x86_64` tag describes the built platform; it does not mean
manylinux portability. Installation examples use source or local artifacts.

Strongest inherited qualification: Linux x86_64, Python 3.13.15, NumPy 2.5.3,
GNU 16.2.1, CPU and a tested AVX2 profile; additional focused NumPy 2.5.1 and
Clang 22.1.8 evidence. Single-GPU CUDA evidence uses NVIDIA SM89 and toolkit
13.3.73. Current-source Windows/MSVC and other selected platforms remain unqualified.
Declared floors (Python >=3.10, NumPy >=1.21.3, scikit-build-core >=0.11,
pybind11 >=3.0) remain unchanged and actual minimum-version compatibility is
unverified.

## Known limitations

- Inherited full CUDA testing retains 98 actual failures: 76 CUDA cast
  combinations and 22 CUDA fill cases. The full CUDA suite is not all-green.
  Preserve all other exclusions in the coverage matrix.
- Physical NCCL/multi-GPU behavior is unvalidated on the current audited source.
- Python shutdown LeakSanitizer attribution remains `OPEN_UNRESOLVED_BOUNDED`,
  including process exit 23. Leak freedom is not established.
- Historical small-operation CUDA admission overhead is approximately
  10 microseconds on the measured hardware/workloads; no performance guarantee
  follows.
- Higher-order autograd, autocast, public Python streams, multi-node rendezvous
  and sharded checkpoints remain unsupported.

Inherited full CPU results are 6,134 passed / 3,684 skipped; inherited full CUDA
results are 10,024 passed / 98 failed / 78 skipped. These results belong to the
previous campaigns on byte-identical runtime source, not freshly rerun full
campaigns for release presentation work. This candidate makes no production-readiness,
universal-platform or performance-superiority claim. Read
[support and limitations](SUPPORT.md) before selecting a profile.
