# CUDA Runtime

How Orthon executes on NVIDIA GPUs: device context discipline, memory
management, kernel dispatch and its contiguity contract, dtype support, and —
just as important — the exact boundary of what has been validated on the
available hardware. Architectural context is in
[architecture.md](architecture.md); the per-operation support matrix is
[cuda_coverage.md](cuda_coverage.md).

## Compile-time CUDA

CUDA support is a build-time property. `WITH_CUDA` defaults to auto-detection
(`pip install .` uses the toolkit if CMake finds one) and can be forced on or
off; `WITH_NCCL` additionally compiles the NCCL collective backend. All CUDA
code sits behind `#ifdef WITH_CUDA`, and public headers include no CUDA
headers, so CPU-only consumers never acquire a CUDA dependency.

The compiled-in state is queryable — `orthon.cuda_enabled()` and
`orthon.nccl_enabled()` — so a CPU-only build reports itself honestly instead
of failing at the first kernel launch, and CUDA-gated tests can skip cleanly.

## Devices and the DeviceGuard

A tensor's `Device` is `cpu` or `cuda:N`. The string `"cuda"` parses to
`cuda:0`; internal helpers that need to match an existing tensor's placement
pass `Device` objects rather than strings, so a model on `cuda:1` builds its
helper tensors on `cuda:1`.

### Why ambient device state is dangerous

The CUDA runtime keeps a *current device* per thread. `cudaMalloc`,
`cudaMemcpy` within a device, kernel launches and cuBLAS calls all implicitly
target it. Code that works by accident on a single-GPU machine — where the
current device is always the right one — silently allocates, copies or
launches on the wrong GPU the moment a second device exists.

Orthon's rule: **an operation's authoritative device is its operands' own,
never the ambient one.** The mechanism is the RAII `DeviceGuard`
(`include/orthon/device/device_guard.h`):

```cpp
{
    DeviceGuard guard(t.device());   // cuda:1 becomes current
    // allocations, memcpys, kernel launches, cuBLAS calls
}                                    // caller's device restored, even on throw
```

Guards are cheap and re-entrant — a guard whose target is already current
records and restores nothing — and constructing one from a CPU device is a
no-op, so a guard can sit unconditionally at a dispatch point serving both
devices. The destructor never throws; a failure during unwinding is reported
through the `ORTHON_DEBUG_DEVICE` diagnostic channel instead of terminating.

### Exact-device allocation and transfer

- `Storage` routes CUDA allocations to the memory pool **with the tensor's
  actual device**; the pool guards its raw `cudaMalloc` against exactly that
  device. A `cuda:N` tensor cannot end up on `cuda:0` by context accident.
- Host↔device copies guard the side that owns the device pointer.
- `cuda:N → cuda:M` uses `cudaMemcpyPeer`, which names **both** device
  ordinals explicitly — unlike `cudaMemcpy(cudaMemcpyDeviceToDevice)`, which
  describes a copy within whichever single device is ambient. No permanent
  peer-access mappings are installed; a pair the hardware cannot service fails
  with the runtime's own error rather than returning a mislabelled tensor.
- Scalar element access on CUDA (`get_item`, full-index `set_item`) routes one
  element through `cudaMemcpy` under a guard, using the stride-aware offset so
  views address the element their logical index names.

The three native `DeviceGuard` tests that need a second physical GPU skip on
the single-GPU development machine; on the 2x A100 release validation run they
executed against a real second device and passed — see
[Validation scope](#validation-scope).

## Memory

CUDA tensor storage allocates through the same caching pool as CPU storage
(`src/core/memory_pool.cpp`): 64-byte size-class bins, per-device free lists,
configurable cache limit with eviction, and allocation/reuse/current/peak/
cached-byte statistics. In a steady-state training loop, iterations recycle
same-shaped blocks from the free lists instead of calling `cudaMalloc`.

A separate per-device **scratch workspace** — a reusable bump-pointer buffer —
serves transient kernel workspaces; full CUDA reductions draw their partial
buffers from it. Scratch has its own reserved/used/peak statistics and reset
control.

Not implemented, deliberately: block splitting/coalescing and stream-aware
allocation. All work runs on the default stream (async NCCL collectives use a
private stream; see [ddp_cloud_validation.md](ddp_cloud_validation.md)).

## Dispatch and the contiguity contract

Public operators branch on operand device at the op level
(`src/ops/cpu/*.cpp` own the dispatch; `src/ops/cuda/*.cu` own the kernels).
The contract on the CUDA side has one central rule:

**Raw CUDA kernels are contiguous-only. Views are normalized at the public op,
never inside a kernel.**

Every kernel indexes its operands linearly. A strided view reaching one would
be read in storage order — wrong values, no error — so the public op wraps
each operand in `kernel_operand()`: identity for a contiguous tensor,
otherwise one same-device physical linearization performed under an explicit
`NoGradGuard`. The autograd node attached afterwards still references the
**original** tensor, so:

- graph shape describes the caller's program, with no hidden nodes for the
  temporary;
- saved-version mutation watches point at the user's storage, not at a copy;
- a view's gradient still routes back through its own view node in logical
  coordinates.

This normalization covers `add`, `mul`, the public unary ops (`neg`, `relu`,
`sigmoid`, `tanh`, `leaky_relu`, `gelu`, `exp`, `sqrt`, `reciprocal`),
`dropout`, floating `cast`, `fused_linear_relu`, matmul, and reductions
(which linearize non-contiguous inputs before reducing).

Alongside the layout rule, every CUDA wrapper carries an explicit **dtype
guard**. Without one, an unsupported dtype would fall through to the float32
kernel and have its bits reinterpreted — plausible-looking garbage, typed as
the original dtype, with no error. Unsupported dtypes therefore raise.

### In-place operations are excluded from normalization

`add_` / `mul_` require a **contiguous destination** and reject strided ones.
Materializing an in-place destination would mutate a temporary copy and
silently discard the caller's write; supporting strided destinations honestly
requires a stride-aware in-place kernel, which is outside the current design
(kernels do not know about strides). The rejection is pinned by
`tests/python/test_runtime_consistency.py`.

## CUDA dtype support

| dtype | CUDA storage/transfer | CUDA compute |
| --- | --- | --- |
| float32 | yes | full op coverage |
| float16 | yes | unary, add/mul (incl. broadcast backward), reductions/max/argmax, matmul, Conv2d, contiguous copy, cast |
| bfloat16 | yes | same op set as float16 (`__nv_bfloat16`; requires sm_80+ for compute) |
| float64 | yes | storage and transfer only (float64 compute is CPU) |
| int32/int64/uint8/int8/bool | yes | storage, transfer, indexing roles (embedding/argmax/NLL targets) |

Float16/BFloat16 kernels use low-precision storage with **FP32 accumulation**
(shared-memory tiles in matmul/Conv2d, float32 intermediate buffers for
broadcast backward — avoiding `atomicAdd(__half*)`).

Two paths use vendor libraries where the shape and dtype make it worthwhile:
FP16 2-D matmul dispatches to cuBLAS Tensor Core GEMM, and large eligible BF16
2-D GEMMs dispatch to guarded cuBLAS (`CUBLAS_COMPUTE_32F_FAST_16BF`); small
or batched BF16 work stays on the custom tiled kernel below a work threshold.
`orthon._matmul_dispatch_path` and the guard-report helpers expose which path
a call took, which the benchmarks use.

Mixed-precision *training* on top of these kernels — FP32 master weights,
dynamic loss scaling — is Python-side machinery described in
[training_stack.md](training_stack.md#mixed-precision).

## Kernel inventory

Handwritten kernels cover: elementwise unary/binary with broadcasting
(forward and backward), reductions (`sum`/`mean`, full and per-dim, using
scratch partials), `max`/`argmax` (two-pass global and per-output-element dim
reduction, scatter backward), 2-D and batched 3-D matmul (tiled shared-memory,
FP32/FP16/BF16), Conv2d (tiled forward; specialized 3×3 stride-1 pad-1
grad-input and grouped grad-weight defaults with environment-variable
selectors — see [nsight_profiling.md](nsight_profiling.md)), MaxPool2d,
softmax and log-softmax, fused cross-entropy, NLL loss, embedding gather with
`atomicAdd` scatter backward, dropout (on-device stateless SplitMix64-style
mask generation), dtype casts, contiguity gather, unbroadcast, and the fused
linear+ReLU prototype.

## Validation scope

Two validation environments, kept distinct:

**Proven on the local single-GPU machine** (RTX 4070 Laptop, Windows/MSVC,
CUDA 13.2 — the ongoing development matrix, measured before the final release
corrections):

- the full CUDA Python suite — 3397 passed, 60 skipped, 0 failed;
- the native CUDA CTest suite — 191 discovered, 188 passed, 3 skipped;
- every CUDA kernel path above, against PyTorch CUDA references;
- device-context handling as far as one device can exercise it: guards
  restore correctly, allocations and copies land on the guarded device,
  `cuda:0` addressing round-trips.

**Proven on the release validation machine** (Linux, 2x NVIDIA
A100-SXM4-80GB, driver 580.126.16, CUDA 12.8, GCC 13.3, NCCL 2.27.3 — a
one-time run of the release-candidate tree as it stood immediately before the
final leaf-gradient-ownership correction, which is device-agnostic and left
every path listed here byte-identical; see
[ddp_cloud_validation.md](ddp_cloud_validation.md)):

- the full CUDA Python suite — 3426 passed, 20 skipped, 0 failed (the
  remaining skips are `>=4`/`8`-GPU and configuration gates);
- the native CUDA CTest suite — 186 discovered, 186 passed, 0 skipped,
  including the three `DeviceGuard` tests that select a second device,
  nested-guard restoration and exception unwinding across devices;
- exact `cuda:0` and `cuda:1` behaviour: construction, allocation,
  arithmetic, views, autograd and model movement on each device, with
  allocations physically landing on the named GPU;
- `cudaMemcpyPeer` across distinct ordinals for real — `Tensor.to` in both
  directions between the two GPUs for float32, float16 and bfloat16, values
  bit-exact, sources preserved;
- cross-GPU autograd in both directions, gradients returning to the source
  tensor's device;
- checkpoint `map_location` to `cuda:0` and to `cuda:1`;
- memory-pool device isolation — a block cached for one GPU is never handed
  to an allocation on the other;
- the NCCL backend at runtime: communicator init and teardown, broadcast,
  all-reduce, async collectives, barrier, and the rejection paths, on real
  GPU tensors across two ranks;
- 2-rank DDP training — blocking, bucketed and overlap modes, including
  multi-step overlap with plain `optimizer.zero_grad()` between steps — with
  gradients and parameters verified synchronized across ranks and clean
  shutdown.

That machine is a **correctness** environment and nothing else. No benchmark
was run on it, and **no published timing is attributed to the 2x
A100-SXM4-80GB correctness machine.**

Performance provenance, stated exactly:

- **Current performance** — the Orthon-vs-PyTorch comparison and every number
  quoted in the README — was measured on the single-GPU RTX 4070 Laptop
  development machine
  ([profiles/PYTORCH_COMPARISON.md](../profiles/PYTORCH_COMPARISON.md)).
- **Historical performance** — [profiles/RESULTS.md](../profiles/RESULTS.md)
  keeps a small set of clearly labelled rows from an older single
  A100-SXM4-40GB cloud machine, measured against an earlier project state.
  They are retained as history and drive no current claim.
- The A100-SXM4-40GB historical benchmark machine and the 2x A100-SXM4-80GB
  correctness machine are different systems and are never conflated.

**Not proven anywhere on the final tree:**

- **`>=4`- and `>=8`-GPU behaviour.** Those test gates skip on the two-GPU
  machine. An earlier project state has a recorded 2/4/8-GPU cloud run — see
  [ddp_cloud_validation.md](ddp_cloud_validation.md) — but the final
  corrected tree itself was revalidated on two GPUs only. Nothing in the
  README or these documents claims the final 0.1.0 tree was validated on
  four or eight GPUs.

There is no CUDA job in CI: GitHub-hosted runners have no GPU, and a green
"CUDA" job that cannot launch a kernel would prove nothing. See
[testing.md](testing.md#cuda-and-multi-gpu-boundary).

## Limitations

Current, deliberate, and stated in current-source terms:

- **Partial CUDA `__setitem__` is unsupported.** `t[prefix] = value` on a CUDA
  tensor raises `"__setitem__: partial CUDA assignment is not supported"`
  before any sub-view is built. Full-index scalar assignment works. Lifting
  this requires a stride-aware CUDA fill kernel.
- **Strided CUDA in-place destinations are unsupported.** `add_`/`mul_`
  require contiguous destinations, as above.
- **Kernels operate on contiguous operands** by design; public ops normalize
  where supported, and the raw `cuda_*` entry points keep contiguous-only
  preconditions.
- **`softmax`/`log_softmax` are contiguous-float32 operations.** Both take an
  explicit `dim` (default last, negative dims supported) and are
  differentiable on CPU and CUDA; other dtypes and strided inputs raise.
  `softmax` keeps a dedicated 2-D last-dimension fast path and computes other
  supported ranks/dims through the log-softmax machinery internally.
- **CUDA `cast` covers Float32 ↔ Float16 ↔ BFloat16 only**; integer/bool casts
  on CUDA raise.
- **CUDA embedding and NLL loss assume in-range indices** (CPU validates
  ranges; the CUDA kernels do not).
- **Everything runs on the default stream**; there is no stream pool or
  compute/copy overlap outside the NCCL async path.
