# Orthon 1.0.0rc1 support and limitations

Orthon 1.0.0rc1 is a prerelease intended for developer evaluation. It is not a
stable or production-qualified release. It makes no universal platform or
performance-superiority claim. Qualification beyond the scoped environments
below remains open. PyPI publication and stable 1.0.0 support are outside this
RC1 scope.

## Qualification scope

| Profile | Evidence and limit |
| --- | --- |
| Linux x86_64 / Python 3.13.15 | Strongest audited environment; GNU 16.2.1, NumPy 2.5.3 |
| NumPy 2.5.1 | Additional focused compatibility testing; no general version-range proof |
| CPU and optional AVX2 | CPU full-suite evidence; scoped AVX2 tests on capable hardware |
| Clang 22.1.8 | Scoped native and focused Python tests using libstdc++; no full Clang qualification |
| Single NVIDIA SM89 GPU | RTX 4070 Laptop GPU, CUDA toolkit 13.3.73; scoped supported profiles |
| Windows | Intended CI/developer workflow; current audited source not locally qualified |
| macOS, Linux ARM64, other Python/compiler/toolkit/GPU versions | Unverified for this release candidate |
| Physical NCCL / multi-GPU | Implemented conditionally; current audited source unvalidated on physical multi-GPU hardware |

Declared package requirements remain Python >=3.10, NumPy >=1.21.3,
scikit-build-core >=0.11 and pybind11 >=3.0. Actual minimum-version compatibility
remains unverified. Current package builds use scikit-build-core 1.1.1 and
pybind11 3.1.0; these passes do not qualify their declared floors.

The CPU Linux wheel is `cp313-cp313-linux_x86_64`, with AVX2 disabled.
It is a native platform wheel, not a portable manylinux distribution. Source
installation requires a compatible native toolchain. CUDA and NCCL require
separate builds, dependencies and actual usable hardware.

## Implemented components

The [audited operator/dtype matrix](docs/cuda_coverage.md) defines the supported
profiles. Orthon implements tensor storage and views, CPU/optional CUDA dispatch,
first-order reverse-mode autograd, NN modules, SGD/Adam, explicit mixed precision,
local multiprocessing collectives, optional single-node NCCL and pickle-free
checkpointing. The [API](docs/api.md), [training stack](docs/training_stack.md) and
[distributed contracts](docs/distributed.md) explain component boundaries.

CUDA compute is partial. Cast conversion is supported for F32/F16/BF16 pairs;
CUDA F64/integer/bool conversion pairs and CUDA `fill_` remain unsupported.
Other exclusions in the matrix remain in force, including F32-only pooling,
softmax/loss/embedding routes, restricted in-place arithmetic, partial CUDA
`__setitem__` and strided CUDA in-place destinations. Allocation/transfer does
not imply compute or backward support. BF16 capability checks still apply.
Higher-order autograd, autocast, public Python streams, multi-node rendezvous,
sharded checkpoints and arbitrary batch-broadcast matmul are unsupported.

## Inherited validation and unresolved findings

The previous full CPU campaign recorded 6,134 passed and 3,684 skipped.
The previous full CUDA campaign recorded 10,024 passed, **98 failed** and
78 skipped: 76 CUDA cast combinations and 22 CUDA fill cases. These are actual
failures, not hidden xfails, and the full CUDA suite is not all-green.
These campaigns are inherited results for byte-identical runtime source; they
were not rerun as RC1 full campaigns. Focused packaging and execution checks
do not extend that full-suite qualification.

Physical NCCL/multi-GPU behavior remains unvalidated on the current audited
source. Local host-transport DDP and simulated transport tests do not substitute
for a real NCCL-enabled build and multi-GPU run.

Python shutdown LeakSanitizer attribution remains `OPEN_UNRESOLVED_BOUNDED`.
Ordinary sanitized CPU Python lifecycles retained process exit 23 and reports of
750,619 bytes in 867 allocations. Bounded lifecycle observations do not establish
final retaining-owner attribution or leak freedom; ordinary functional passes
are not LeakSanitizer passes. Independent disposition remains required.

Historical admission measurements on the SM89 stack showed approximately
10 microseconds of added preflight cost for small CUDA embedding/NLL operations:
embedding 4.568 to 14.503 microseconds; NLL forward 5.053 to 15.099 and backward
14.737 to 25.276 microseconds. The tiny indexed step measured 43.714 to 64.928
microseconds. These are hardware/workload-specific inherited timings, not fresh
RC1 performance measurements or a general performance guarantee.

The checkpoint wire identifier remains `ml-runtime-checkpoint`, version 1.
Release package versioning does not migrate that format. See the
[format specification](docs/checkpoint_format.md).

## Remaining qualification

Before broader support or a stable release, independently review minimum-version
jobs, Windows and other selected platforms, physical NCCL/multi-GPU execution,
shutdown attribution, and workload-specific latency acceptance. Select the
promised support scope from evidence and disclose any remaining unsupported
profiles.
