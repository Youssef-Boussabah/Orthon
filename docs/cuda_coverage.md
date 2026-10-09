# CPU and CUDA coverage

This matrix describes supported profiles in the current tree, not API parity with
PyTorch. The schemas/constraints in `src/dispatch/builtin_registration.cpp`, native
semantic guards and executable specs define behavior. Compilation flags alone do
not validate physical execution.

Abbreviations: F32=float32, F64=float64, F16=float16, BF16=bfloat16.
The nine storage dtypes are F32/F64/F16/BF16/int32/int64/uint8/int8/bool.

Physical bytes and mutation generation belong to Buffer/Allocation. Runtime
contexts provide a size-class MemoryPool and ScratchArena/ScratchLease services;
CUDA cache and scratch retirement/reuse respect CompletionToken ordering.
See [CUDA runtime](cuda_runtime.md) for memory and stream ownership.

| Family | CPU | CUDA | Boundary |
| --- | --- | --- | --- |
| `empty`, `zeros`, `ones`, `full`, `from_numpy` | Nine dtypes | Create on CPU, then `to()` | No device keyword on Python factories; `from_numpy` requires C-contiguous input; BF16 override accepts F32 input |
| Transfers and host inspection | Nine dtypes | Nine contiguous storage dtypes | Strided CUDA materialization is F32/F16/BF16 only |
| View metadata: reshape/transpose/slice/index | Shared Buffer | Shared Buffer | Legal geometry required; Python slice objects/advanced indexing unsupported |
| `contiguous` | Nine dtypes | F32/F16/BF16 strided packing | Contiguous no-op remains possible for other dtypes |
| Cast | All dtype pairs | F32/F16/BF16 conversion pairs | Same-dtype cast normalizes contiguity; CUDA F64/integer/bool conversion pairs unsupported |
| Add/mul, composed sub/div and scalar arithmetic | F32/F64 | F32/F16/BF16 | Broadcasting supported where validated; division composes reciprocal/multiply |
| Neg/relu/sigmoid/tanh/leaky_relu/gelu/exp/sqrt/reciprocal | F32/F64 | F32/F16/BF16 | Grad accumulation must also support the profile |
| Sum/mean/max/argmax | F32/F64 | F32/F16/BF16 | Optional one dimension, keepdim; argmax output int64; extrema follow first-NaN/first-maximum policy |
| Matmul | F32/F64 | F32/F16/BF16 | 2-D or equal-batch 3-D; BF16 capability guards apply |
| Conv2d | F32 | F32/F16/BF16 | NCHW, stride/padding; low-precision CUDA accumulation uses F32 |
| MaxPool2d | F32 | F32 | No general low-precision pool route |
| Softmax/log_softmax | F32 | F32 | Contiguous input, normalized dimension; dedicated row fast path where applicable |
| Cross entropy / NLL loss | F32 | F32 | `[N,C]`; NLL targets int32/int64, mean/sum/none; valid CUDA class indices required |
| Embedding | F32 weights | F32 weights | Integer indices, optional padding index; no sparse optimizer contract |
| Fused Linear+ReLU | F32 | F32 | 2-D input and 1-D bias; backward composes canonical ops |
| Dropout mask | F32 | F32 | Seeded deterministic mask; module eval is identity |
| LayerNorm, BatchNorm, attention, Transformer blocks | Composed supported primitives | Composed supported primitives | No fused norm/FlashAttention claim; constituent profiles limit dtype coverage |
| `fill_` and scalar `get_item`/`set_item` | F32 | Scalar access F32; `fill_` unsupported | Use CPU factory then transfer for CUDA initialization |
| `add_`, `mul_` | F32/F64 | F32 | CUDA operands/destination contiguous; F16/BF16/F64 in-place arithmetic unsupported |
| Finiteness inspection | Nine dtypes | Nine dtypes | Observation does not imply compute support |
| SGD / bare Adam | Supported update profiles; Adam F32/F64 | Adam F32 | No SGD momentum; AMP uses F32 masters |
| Explicit AMP | F32 pass-through; CPU low precision compute limited | F16/BF16 models with F32 masters | No autocast; contiguous master/model copy contract |
| Checkpoint file APIs | Bit-preserving CPU snapshots | Copy through CPU; restore to supported target | Local single-file format; no sharding or distributed runtime state |
| Local collectives | Host transport | Host staging | Not GPU-resident NCCL evidence |
| Optional NCCL collectives | Unsupported tensors | Contiguous F32/F16/BF16, exact communicator device | Sum only; NCCL DDP averages F32 gradients |

BF16 native kernels require suitable GPU capability (sm_80+). Planner capability
checks further constrain cuBLAS selection. A supported allocation or transfer
must not be mistaken for supported compute or backward.

## Known unsupported behavior

Higher-order autograd, autocast, public Python stream control, multi-node
rendezvous, sharded checkpoints, arbitrary batch-broadcast matmul, partial CUDA
`__setitem__` and strided CUDA in-place destinations are unsupported.
CUDA cast pairs outside the compute trio and CUDA `fill_` account for known
specification failures; exact environment-specific counts belong in validation
reports rather than this matrix. Tests do not hide them with xfail.

## Validation

Tests gate physical CUDA and multi-device cases on usable hardware. Build flags
alone do not establish runtime coverage. See [testing](testing.md) for profiles
and [distributed](distributed.md) for NCCL validation requirements.
