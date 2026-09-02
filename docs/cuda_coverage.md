# CUDA Coverage Matrix

Per-operation support across the public Python API: does it run on CPU, on
CUDA, with which dtypes, with autograd, and under which restrictions. This is
a framework capability matrix, not a PyTorch parity claim. The execution model
behind the CUDA column — dispatch, contiguity contract, `DeviceGuard`, memory
pool — is explained in [cuda_runtime.md](cuda_runtime.md); autograd behaviour
in [autograd.md](autograd.md).

Status meanings:

- `complete`: CPU/CUDA forward and backward are implemented and tested for the
  supported API scope.
- `partial`: useful today, but limited by implementation scope, host
  materialization, or indirect backward behavior.

Distributed note: `orthon.distributed` has a correctness-first local backend
and a limited single-node NCCL backend; its exact supported-versus-validated
contract is in [ddp_cloud_validation.md](ddp_cloud_validation.md).

## The matrix

| API name | CPU forward | CUDA forward | CPU backward | CUDA backward | Tests | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `empty`, `zeros`, `ones`, `full` | yes | via `.to("cuda")` | n/a | n/a | yes | partial | CPU-first creation with optional `dtype=` keyword. All nine dtypes (Float32, Float64, Float16, BFloat16, Int32, Int64, UInt8, Int8, Bool) are supported for allocation and zero-fill; `full` uses bit-conversion helpers for Float16/BFloat16. CUDA tensors created by transfer. |
| `from_numpy` | yes | via `.to("cuda")` | n/a | n/a | yes | partial | Accepts all nine dtypes via numpy `kind`+`itemsize` dispatch. Float16 bits are copied directly (IEEE 754 layout matches `__half`). Float64 CPU compute and Float16/BFloat16 CUDA compute are supported; Int/Bool remain storage-only. |
| `Tensor.to`, `orthon.to` | yes | yes | yes (ToNode) | yes (ToNode) | yes | complete | Device movement supports CPU/CUDA transfer for all nine supported dtypes; mixed-device ops raise. Non-contiguous CPU copies use a dtype-generic `char*`+`dtype_size` walk. A `cuda:N` -> `cuda:M` transfer uses `cudaMemcpyPeer`, which names both device ordinals explicitly, rather than an ambient-context device-to-device `cudaMemcpy`; validated on real 2x A100 hardware in both directions for float32/float16/bfloat16 (final release validation — see [cuda_runtime.md](cuda_runtime.md#validation-scope)). No peer-access mappings are installed; the copy is correct, not peer-optimized. On single-GPU machines the multi-GPU tests skip honestly. |
| `cuda_synchronize` | n/a | yes | n/a | n/a | yes through CUDA tests | complete | Synchronization helper for timing/profiling. |
| `add`, `Tensor.__add__`, `Tensor.add_` | yes | yes | yes | yes | yes | complete | Broadcasting and CUDA backward are covered for current tensor shapes. |
| `mul`, `Tensor.__mul__`, `Tensor.mul_` | yes | yes | yes | yes | yes | complete | Used by optimizers and composite layer backward paths. |
| `sub`, `div`, scalar arithmetic | yes | yes | yes | yes | yes | complete | `sub`/`div` compose `add`+`neg` and `mul`+`reciprocal`, inheriting their validation and autograd. Python scalars on either side (`x * 2`, `2 - x`, `4 / x`) become rank-0 constants on the tensor's exact device and dtype, with value-insensitive scalar autograd nodes. |
| `neg`, unary `-` | yes | yes | yes | yes | yes | complete | Differentiable on both devices through the shared activation node; value-insensitive (backward is `-grad`, so the input payload is never read or watched). CUDA forward supports Float32/Float16/BFloat16. |
| `sqrt` | yes | yes | yes | yes | yes | complete | Used by Adam and composite normalization layers; large-tensor CUDA parity tests included. |
| `reciprocal` | yes | yes | yes | yes | yes | complete | Used by Adam denominator updates and composite normalization layers. |
| `relu`, `Tensor.relu`, `functional.relu`, `nn.ReLU` | yes | yes | yes | yes | yes | complete | Baseline activation. |
| `sigmoid`, `Tensor.sigmoid`, `functional.sigmoid`, `nn.Sigmoid` | yes | yes | yes | yes | yes | complete | CPU/CUDA forward and backward parity-tested against PyTorch. |
| `tanh`, `Tensor.tanh`, `functional.tanh`, `nn.Tanh` | yes | yes | yes | yes | yes | complete | Both paths support non-contiguous input: the CUDA path materialises only the kernel operand and keeps the autograd node on the original tensor. |
| `leaky_relu`, `Tensor.leaky_relu`, `functional.leaky_relu`, `nn.LeakyReLU` | yes | yes | yes | yes | yes | complete | Configurable `negative_slope`. The CUDA path accepts non-contiguous input, like every other public unary. |
| `gelu`, `Tensor.gelu`, `functional.gelu`, `nn.GELU` | yes | yes | yes | yes | yes | complete | Exact GELU, matching PyTorch `approximate="none"` within float32 tolerance. |
| `exp` | yes | yes | yes | yes | yes | complete | Composes differentiable softmax weights in attention as `exp(log_softmax(scores))`. |
| `sum`, `Tensor.sum`, `functional.sum` | yes | yes | yes | yes | yes | complete | Tensor-valued scalars for full reductions; supports `dim`, negative dims, `keepdim`. Non-contiguous CUDA inputs are made contiguous before reduction. |
| `mean`, `Tensor.mean`, `functional.mean` | yes | yes | yes | yes | yes | complete | Same reduction scope as `sum`; backward divides by the reduced element count. |
| `max`, `Tensor.max`, `functional.max`, `orthon.max` | yes | yes | yes | yes | yes | complete | Full/dim reduction; optional `dim`, negative dims, `keepdim`. Gradient scatters to the argmax position (no tie-sharing). CPU: stride-based walk. CUDA: contiguous copy then two-pass global or per-output-element kernel. |
| `argmax`, `Tensor.argmax`, `functional.argmax`, `orthon.argmax` | yes | yes | n/a | n/a | yes | complete | Returns Int64 index tensor; no autograd. Same kernel path as `max`; only the index half of the pair is returned. Tie policy: first-encountered element wins. |
| `reshape`, `Tensor.reshape` | yes | yes | yes | yes | yes | complete | Metadata/view-style op with reshape backward. |
| `transpose`, `Tensor.transpose` | yes | yes | yes | yes | yes | complete | O(1) stride-swap view; backward routes the gradient back through `TransposeNode` (itself a view, no data movement). Covered by dedicated CPU/CUDA backward tests (`test_contiguous_autograd.py`) plus view/composition gradient suites on both devices. |
| `flatten`, `Tensor.flatten`, `functional.flatten`, `nn.Flatten` | yes | yes | yes | yes | yes | complete | Implemented through reshape, so autograd/device behavior follows reshape. |
| `matmul`, `Tensor.__matmul__`, `Tensor.matmul` | yes | yes | yes | yes | yes | complete | CPU path supports Float32 and Float64. CUDA path expects contiguous 2-D inputs and supports Float32, Float16, and BFloat16 (FP16/BF16 storage, FP32 tile accumulation); FP16 2-D matmul dispatches through cuBLAS Tensor Core GEMM, and large eligible BF16 2-D GEMMs can dispatch to the guarded cuBLAS path. |
| Batched 3D `matmul` | yes | yes | yes | yes | yes | partial | Supports `[B, M, K] @ [B, K, N]` for attention. Not a full NumPy/PyTorch batched-matmul generalization. CUDA materializes non-contiguous inputs before launch while preserving the autograd graph. |
| `nn.Linear` | yes | yes | yes | yes | yes | complete | Uses matmul/add; covered in CPU and CUDA graph tests. |
| `conv2d`, `nn.Conv2d` | yes | yes | yes | yes | yes | complete | CUDA optimized defaults (tiled forward, 3x3 grad_input, grouped3x3/fused grad_weight+bias) are Float32-only. Float16 and BFloat16 CUDA use generic kernels (FP16/BF16 storage, FP32 accumulation) for all forward and backward paths. |
| `maxpool2d`, `nn.MaxPool2d` | yes | yes | yes | yes | yes | complete | Forward/backward CUDA tests exist. |
| `nn.Dropout` | yes | yes | yes via saved scaled mask | yes via saved scaled mask | yes | complete | Inverted dropout with native CPU/CUDA mask generation. CUDA masks are generated on device with a stateless SplitMix64-style RNG; no global RNG system. |
| `nn.BatchNorm1d`, `nn.BatchNorm2d` | yes | yes | yes | yes | yes | complete | Composition-based over Tensor reductions and elementwise ops on the input's own device; running statistics are detached via explicit no-grad same-device snapshots (no host round-trip). Parity-tested on CPU (including float64) and CUDA (including float16/bfloat16). Scope: `BatchNorm1d` takes `[N, C]`, `BatchNorm2d` takes `[N, C, H, W]`. No dedicated fused BatchNorm kernel. |
| `nn.LayerNorm`, `functional.layer_norm`, `orthon.layer_norm` | yes | yes | yes | yes | yes | complete | Composition over tensor reductions and elementwise ops; no fused CUDA LayerNorm kernel. |
| `nn.Embedding`, `functional.embedding`, `orthon.embedding` | yes | yes | n/a for indices, yes for weight | n/a for indices, yes for weight | yes | complete | Gathers int32/int64 token IDs and accumulates weight gradients, with repeated-index accumulation and `padding_idx` gradient skipping. CUDA backward uses `atomicAdd` scatter-add; CUDA indices are expected to be in range. `indices` and `weight` must be contiguous, enforced identically on CPU and CUDA — both are read with linear row-major indexing, so accepting strided indices on one device would silently return storage-order rows. |
| `functional.scaled_dot_product_attention`, `orthon.scaled_dot_product_attention` | yes | yes | yes | yes | yes | partial | Composition-based single-head `[B, T, D]` attention using batched matmul, `log_softmax` + `exp`, optional additive mask, and optional dropout. No fused/FlashAttention kernel. |
| `nn.SelfAttention` | yes | yes | yes | yes | yes | complete | Single-head self-attention module with q/k/v/out projections. |
| `nn.MultiHeadSelfAttention` | yes | yes | yes | yes | yes | complete | Multi-head self-attention with per-head projections. Uses 3D batched matmul and `contiguous()` after transpose to feed reshape. CUDA backward tested end-to-end. |
| `Tensor.contiguous`, `orthon.contiguous` | yes | yes | yes | yes | yes | complete | Materializes non-contiguous views into contiguous storage while preserving the autograd graph. CPU stride-aware walk is dtype-generic (all nine dtypes); CUDA uses the gather kernel. Backward is an identity pass-through; upstream nodes handle gradient transposition. |
| `nn.TransformerEncoderBlock` | yes | yes | yes | yes | yes | complete | Pre-norm encoder block composed from LayerNorm, MultiHeadSelfAttention, Dropout, Linear, and GELU. Composition-based; no fused Transformer kernel, FlashAttention, or fused LayerNorm. |
| `fused_linear_relu`, `nn.FusedLinearReLU` | yes | yes | yes | yes indirectly | yes | partial | Forward is fused; backward reuses existing matmul/ReLU/broadcast-gradient paths rather than a custom fused kernel. Scope: float32, 2-D inputs, 1-D bias. |
| `softmax` | yes | yes | yes | yes | yes | complete | Explicit `dim` argument (default `-1`, negative dims normalized), rank >= 1, contiguous float32. Differentiable on both devices: attaches one public `SoftmaxNode`, whose backward is composed from existing primitives. The 2-D last-dimension case keeps a dedicated row-wise fast path (fused CUDA kernel, direct CPU loop); other ranks/dims compute values via `log_softmax` + `exp` internally under no-grad without exposing that decomposition in the graph. |
| `log_softmax` | yes | yes | yes | yes | yes | complete | Contiguous float32 tensors with explicit `dim` and negative dims. |
| `nll_loss`, `nn.NLLLoss` | yes | yes | yes | yes | yes | complete | `[N, C]` log-probabilities, int32/int64 class-index targets, `mean`/`sum`/`none` reductions. CPU validates target ranges; CUDA assumes indices are in range. |
| `cross_entropy` | yes | yes | yes | yes | yes | complete | Fused logits + class-index target loss. Equivalent to `nll_loss(log_softmax(logits, dim=-1), targets)` for the supported `[N, C]` case, kept fused for training-path speed. |
| `nn.Module`, `nn.Parameter`, `nn.Sequential` | yes | device moves supported | n/a | n/a | yes | complete | Module tree and parameter registration. `Module.cast(dtype)`, `Module.half()`, and `Module.bfloat16()` recursively convert parameters in-place while preserving `requires_grad`. |
| `orthon.cast`, `Tensor.cast` | yes | yes (Float32↔Float16↔BFloat16) | yes (CastNode) | yes (CastNode) | yes | partial | CPU converts between any dtype pair via double as intermediate; CUDA covers the three floating pairs. Floating→floating conversion is differentiable (CastNode; backward casts the gradient back to the source dtype); conversions involving integer/bool dtypes never join a graph, and a same-dtype cast is a contiguity normalization, never a CastNode. |
| `amp.MixedPrecisionOptimizer` | yes | yes | n/a | n/a | yes | complete | Wraps any optimizer with FP32 master weights. Low-precision (FP16/BF16) params are upcast to FP32 for the optimizer step, then cast back. FP32 params pass through unchanged. |
| `amp.GradScaler` | yes | yes | n/a | n/a | yes | partial | Dynamic loss scaling: scale, idempotent unscale with finiteness check, step skipping on non-finite gradients, backoff/growth. Disabled mode is the BF16-safe path. No autocast. |
| `optim.SGD` | yes | yes | n/a | n/a | yes | complete | Decoupled weight decay; no momentum. |
| `optim.Adam` | yes | yes | n/a | n/a | yes | complete | Bias-corrected moments, decoupled weight decay; large-shape CPU/CUDA parity coverage. |
| `no_grad`, `set_grad_enabled`, `is_grad_enabled` | yes | yes | n/a | n/a | yes | complete | Graph suppression works for the CPU/CUDA paths covered by tests. |
| `state_dict`, `load_state_dict`, `save_state_dict` | yes | CPU materialization | n/a | n/a | yes | partial | One pickle-free archive; tensors are stored as raw CPU bytes taken under an explicit no_grad `to("cpu")` snapshot, so a CUDA tensor serializes exactly and reloads onto the destination's devices. |
| `save_checkpoint`, `load_checkpoint` | yes | CUDA-safe load into existing CUDA params | n/a | n/a | yes | partial | Versioned pickle-free archive for model state, optimizer state, and metadata: bit-exact round-trip, Adam/SGD state restore, `map_location`, resume-training equivalence. Not sharded, distributed, or cloud-native. |
| Memory pool controls | yes | yes | n/a | n/a | yes | partial | 64-byte size-class bins, configurable cache limits with eviction, allocation/reuse/current/peak/cached-byte stats; separate per-device bump-pointer scratch workspaces with reserved/used/peak stats. Full CUDA reductions use scratch for partial buffers. No block splitting/coalescing or stream-aware CUDA allocator behavior. |

## Dtype support summary

- The `DType` enum has nine values: Float32, Float64, Float16, BFloat16,
  Int32, Int64, UInt8, Int8, Bool. All nine are supported for allocation and
  fill (`empty`/`zeros`/`ones`/`full` with a `dtype=` keyword), `from_numpy`,
  CPU↔CUDA transfer, `contiguous()`, `item()` and `tolist()`. `fill_` remains
  float32-only.
- **Float64 compute is CPU:** unary, binary, reductions, max, matmul and their
  backward paths are templatized over Float32/Float64 and parity-tested
  against PyTorch float64 (`tests/python/test_float64.py`).
- **Float16 and BFloat16 compute is CUDA:** unary, add/mul (including
  broadcast backward), reductions/max/argmax, matmul (2-D and batched),
  contiguous copy, cast and Conv2d forward/backward, all with low-precision
  storage and FP32 accumulation. BFloat16 kernels use `__nv_bfloat16` and
  require sm_80+. Broadcast backward accumulates into a float32 intermediate
  (avoiding `atomicAdd(__half*)`). Parity suites:
  `test_float16_cuda.py`, `test_bfloat16_cuda.py`, `test_low_precision.py`.
- The AMP stack over these dtypes (master weights, dynamic scaling) is
  described in [training_stack.md](training_stack.md#mixed-precision).

## Runtime consistency contract

The device an operation runs on does not change what it means. No CUDA kernel
is taught about strides; normalization happens at the public op
([cuda_runtime.md](cuda_runtime.md#dispatch-and-the-contiguity-contract)
explains the mechanism). The load-bearing points, pinned by
`tests/python/test_runtime_consistency.py`:

- **CUDA view consumption.** `add`, `mul`, all nine public unaries, `dropout`,
  real-dtype `cast` and `fused_linear_relu` accept the same non-contiguous
  inputs their CPU forms accept, via the shared `kernel_operand()` helper; the
  autograd node stays attached to the original tensor.
- **Guards on every CUDA wrapper.** Each carries an explicit dtype guard (an
  unguarded fall-through would reinterpret bits) and layout handling (an
  unguarded view would be read in storage order).
- **In-place ops deliberately excluded.** `add_`/`mul_` require a contiguous
  destination: materializing a destination would mutate a temporary and drop
  the caller's write.
- **Same-dtype `cast` is one contract on both devices** — identity for a
  contiguous source, `ContiguousNode` for a strided one, never a `CastNode`.
  Callers wanting an independent physical copy use `x.to(x.device)`.
- **Exact CUDA device propagation.** Internal helpers move tensors with
  `Device` objects, never the string `"cuda"` (which parses to `cuda:0`), so
  a module on `cuda:1` builds its helper tensors on `cuda:1`.
- **CUDA scalar element access is safe.** `get_item`, full-index `set_item`
  route one element through `cudaMemcpy` under a `DeviceGuard` with
  stride-aware offsets; a successful write bumps the storage version exactly
  once. Partial CUDA `__setitem__` raises before any sub-view is built.
- **Device memory never routes through the CPU `Allocator`.** `Storage` sends
  CUDA allocation straight to the memory pool with the tensor's actual
  device, so a `cuda:N` tensor cannot be allocated on `cuda:0`.

## Known gaps

Current highest-value framework gaps, in current-state terms:

1. **Fused CUDA BatchNorm** — `BatchNorm1d`/`BatchNorm2d` are
   composition-based over reductions and elementwise ops (numerically complete
   on both devices); a dedicated fused kernel with saved training statistics
   would reduce launch count.
2. **A full RNG API** — dropout supports per-module deterministic seeding and
   one framework stream drives initialization, but there is no torch-style
   generator object, stream-aware state, or stochastic decoding (temperature,
   top-k, nucleus, beam search). Greedy argmax generation is native.
3. **Fused CUDA LayerNorm** — LayerNorm is composition-based; a fused kernel
   would cut launch counts in Transformer blocks.
4. **General batched matmul** — the 3-D batched path covers the attention
   shapes only.
