# Python API reference

This reference covers the application API of the current Orthon 1.0.0rc1
development tree. Source and executable tests define behavior. Availability is
bounded by the [CPU/CUDA matrix](cuda_coverage.md); API similarity does not imply
PyTorch parity. Import with `import orthon as o`.

## Root module: orthon

| Surface | Signature / behavior |
| --- | --- |
| Tensor | Native tensor handle, produced by factories/ops; no Python data constructor |
| DType | Enum: Float32, Float64, Float16, BFloat16, Int32, Int64, UInt8, Int8, Bool |
| DeviceType | CPU / CUDA enum |
| Device | `Device()`, `Device(spec)`, `Device(type, index)`; immutable type/index, `is_cpu()`, `is_cuda()`; `"cuda"` means ordinal 0 |
| empty / zeros / ones | `(shape, dtype="float32")`; CPU allocation; empty values uninitialized |
| full | `(shape, value, dtype="float32")`; CPU factory |
| from_numpy | `(array, dtype=None)`; copies C-contiguous supported NumPy arrays; explicit override only BF16 from F32 input |
| fill_ | `(tensor, value)`; CPU F32 write |
| add / mul / sub / div | `(a, b)` Tensor operands; scalar forms use Tensor Python operators |
| neg / relu / sigmoid / tanh / gelu / exp / sqrt / reciprocal | `(tensor)` |
| leaky_relu | `(tensor, negative_slope=0.01)` |
| dropout | `(tensor, p, seed)`; seeded primitive; module controls training/eval |
| reshape | `(tensor, shape)`; supported contiguous reshape geometry |
| flatten | `(tensor, start_dim=0, end_dim=-1)` |
| transpose | `(tensor, dim0, dim1)`; metadata view |
| contiguous | `(tensor)`; materializes logical order when needed |
| to | `(tensor, device)`; independent copy; string or Device |
| cast | `(tensor, dtype)`; dtype string; same-dtype contiguity normalization |
| sum / mean / max / argmax | `(tensor, dim=None, keepdim=False)`; one dimension or all; max returns values, argmax int64 indices |
| matmul | `(a, b)`; 2-D or equal-batch 3-D |
| fused_linear_relu | `(x, weight, bias)`; F32 2-D input, required 1-D bias |
| conv2d | `(input, weight, bias, stride_h=1, stride_w=1, pad_h=0, pad_w=0)`; NCHW, required bias argument |
| maxpool2d | `(input, kernel_h, kernel_w, stride_h, stride_w, pad_h=0, pad_w=0)` |
| softmax / log_softmax | `(tensor, dim=-1)`; contiguous F32 |
| cross_entropy | `(logits, targets)`; `[N,C]` F32 and integer target sequence |
| nll_loss | `(log_probs, targets, reduction="mean")`; target Tensor or sequence; mean/sum/none |
| embedding | `(indices, weight, padding_idx=None)`; indices Tensor; nn.Embedding also accepts Python inputs |
| layer_norm | `(tensor, normalized_shape, weight=None, bias=None, eps=1e-5)` |
| scaled_dot_product_attention | `(q, k, v, mask=None, dropout_p=0.0, training=True)`; compositional attention with additive mask |
| isfinite_all | `(tensor)` -> bool |
| is_grad_enabled | `is_grad_enabled() -> bool`; query thread-local grad mode |
| set_grad_enabled | `set_grad_enabled(enabled) -> None`; set thread-local grad mode |
| no_grad | `()` context manager; restores previous grad mode |
| manual_seed | `(seed)`; framework-local Python initialization/dropout stream |
| cuda_enabled / nccl_enabled | `()` -> compile-feature bool; no hardware guarantee |
| cuda_synchronize | `()`; current physical CUDA device compatibility wait; CPU no-op |
| nn / optim / amp / training | Imported modules documented below |

Max-pooling selects the first maximum in row-major window order, including
infinity ties. NaNs are ignored when a numeric value is present; an all-NaN
window returns the first NaN and routes its gradient there. An admitted window
with no input elements returns negative infinity and contributes no input gradient.

### Tensor properties and methods

Read-only properties: `shape`, `strides`, `storage_offset`, `ndim`, `numel`,
`dtype` (string), `device`, `is_contiguous`, `requires_grad`, `is_leaf`, `grad`.
`numel` also supports call syntax. `tolist()` returns a flat logical value list;
`item()` requires one element. CUDA host reads complete the required transfer.

Use `requires_grad_(value)` to set the flag, `backward(retain_graph=False)` to
seed with ones, `zero_grad()` to clear `.grad`, and `set_grad(grad)` to assign an
accumulator. There is no gradient seed parameter or higher-order backward API.
See [autograd](autograd.md) for accumulation/history boundaries.

Methods mirror `reshape(shape)`, `flatten(start_dim=0, end_dim=-1)`,
`transpose(dim0, dim1)`, `contiguous()`, `to(device)`, `cast(dtype)`,
`sum/mean/max/argmax(dim=None, keepdim=False)`, `neg/relu/sigmoid/tanh/gelu/exp()`,
`leaky_relu(negative_slope=0.01)`, `log_softmax(dim=-1)`, `matmul(b)` and
`isfinite_all()`. `slice(dim, start, end)` is an explicit bounded view, with no
step argument. Integer/tuple/list indexing is supported; Python slice objects,
Ellipsis and advanced indexing are unsupported. Full scalar `get_item(indices)`
and `set_item(indices, value)` are F32; partial CUDA assignment rejects.

`fill_(value)`, `add_(other, alpha=1.0)` and `mul_(alpha)` mutate under their
supported dtype/device profiles. CUDA in-place destinations must be contiguous.
`add_` rejects partially overlapping source/destination ranges before mutation,
including when alpha is zero. Exact aliases, disjoint sibling views and empty
views remain supported and each successful call advances the destination version once.
Python `+`, `-`, `*`, `/` accept Tensor or supported numeric scalar operands;
`@` calls matmul. General NumPy array arithmetic is not an interoperability API.

## Neural networks: orthon.nn

| Class | Constructor / scope |
| --- | --- |
| Module | `Module()`; subclass `forward`; registered parameters/children/buffers |
| Parameter | `Parameter(data)`; marks Tensor trainable; module assignment registers data |
| Linear / FusedLinearReLU | `(in_features, out_features)`; 2-D input; weight and bias always present |
| ReLU / Sigmoid / Tanh / GELU | `()` |
| LeakyReLU | `(negative_slope=0.01)` |
| Flatten | `(start_dim=1, end_dim=-1)`; default differs from root flatten |
| Sequential | `(*modules)` |
| Conv2d | `(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)`; optional bias |
| MaxPool2d | `(kernel_size, stride=None, padding=0)` |
| Dropout | `(p=0.5, seed=None)`; eval identity; seeded training mask |
| BatchNorm1d / BatchNorm2d | `(num_features, eps=1e-5, momentum=0.1, affine=True)`; persistent running mean/variance; composed primitives |
| LayerNorm | `(normalized_shape, eps=1e-5, elementwise_affine=True, device=None)` |
| Embedding | `(num_embeddings, embedding_dim, padding_idx=None, device=None)` |
| NLLLoss | `(reduction="mean")`; call `(log_probs, targets)` |
| SelfAttention | `(embed_dim, dropout=0.0)`; call `(x, mask=None)` |
| MultiHeadSelfAttention | `(embed_dim, num_heads, dropout=0.0)`; call `(x, mask=None)` |
| TransformerEncoderBlock | `(embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0, eps=1e-5)`; call `(x, mask=None)` |

Module methods: `parameters()`, `named_parameters(prefix="")`,
`buffers(recurse=True)`, `named_buffers(prefix="", recurse=True)`,
`register_buffer(name, tensor, persistent=True)`, `train()`, `eval()`, `zero_grad()`,
`to(device)`, `cast(dtype)`, `half()`, `bfloat16()`, `state_dict()` and
`load_state_dict(state, strict=True)`. Parameter Python owners survive supported
representation transforms. State dictionaries contain live references;
files snapshot values. See [training stack](training_stack.md).

## Optimizers: orthon.optim

`SGD(params, lr, weight_decay=0.0, decay_bias=False)` has no momentum or parameter
groups. `Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8,
weight_decay=0.0, decay_bias=False)` supports CPU F32/F64 and CUDA F32 parameters.
Both expose `step()`, `zero_grad()`, `get_lr()`, `set_lr(lr)`, `state_dict()` and
`load_state_dict(state, map_location=None)`. Weight decay is decoupled and skips
one-dimensional values unless decay_bias is enabled. Bindings retain parameter
owners; optimizer-only remapping cannot move them. SGD's direct config-only loader
ignores map_location; file transactions validate placement.

## Explicit AMP: orthon.amp

`MixedPrecisionOptimizer(model, optimizer_cls, **optimizer_kwargs)` provides
`step()`, `zero_grad()`, `sync_to_model()`, `get_lr()`, `set_lr(lr)`,
`state_dict()` and `load_state_dict(state, map_location=None)`. F16/BF16 model
owners have separate F32 masters; F32 owners pass through. Built-in SGD/Adam
provide the private reconciliation protocol. There is no autocast API.

`GradScaler(init_scale=1024.0, growth_factor=2.0, backoff_factor=0.5,
growth_interval=2000, enabled=True)` exposes `scale(loss)`, `unscale_(mp_optimizer)`,
`step(mp_optimizer)`, `update()`, `state_dict()`, `load_state_dict(state)` and
read-only `scale_factor`, `enabled`, `found_inf`. Adaptation occurs in step;
update resets cycle state. See [training](training_stack.md) for CUDA and DDP use.

## File checkpoint helpers

These root names delegate to the checkpoint facade:

| API | Signature |
| --- | --- |
| save_state_dict | `(model, path)` |
| load_state_dict | `(model, path, map_location=None)` |
| save_checkpoint | `(path, model=None, optimizer=None, metadata=None)` |
| load_checkpoint | `(path, model=None, optimizer=None, map_location=None)` -> metadata dict |

Paths accept filesystem path objects. No pickle is loaded. The single-file format
stores model/optimizer/metadata, not distributed runtime, scaler or RNG state
implicitly. See [checkpoint format](checkpoint_format.md).

## Training helpers: orthon.training

`classification_accuracy(logits, targets)` returns a scalar accuracy.
`train_step(model, batch_x, batch_y, loss_fn, optimizer)` and
`eval_step(model, batch_x, batch_y, loss_fn)` return loss/accuracy/num_examples.
Evaluation sets eval mode; use no_grad explicitly when desired.
`weighted_metric_average(step_outputs)` weights metrics by example count and
rejects zero total examples.

## Distributed: orthon.distributed

Import this module explicitly; it is not part of the root's eagerly imported
module inventory.

| API | Signature / behavior |
| --- | --- |
| Backend | Public abstract transport extension: rank/world_size properties, all_reduce/broadcast/barrier/shutdown |
| register_backend | `(name, factory)` |
| rank_config_from_env | `(environ=None)` -> rank/world_size/local_rank/local_world_size/backend dictionary |
| init_process_group | `(backend=None, **kwargs)` -> Backend; creates captured group |
| destroy_process_group | `()`; shuts down active group |
| is_initialized / get_rank / get_world_size | `()` |
| all_reduce | `(tensor, op="sum", async_op=False)`; in-place tensor or Work |
| broadcast | `(tensor, src=0, async_op=False)`; in-place tensor or Work |
| barrier | `()` |
| spawn | `(fn, args=(), nprocs=2, backend="local")`; rank-ordered local worker results |
| DistributedDataParallel / DDP | `(module, src=0, sync_grads=True, bucket_size_mb=None, overlap_grad_sync=False)` |
| Work | Transport-returned handle; `wait()`, `is_completed()` |

DDP delegates call/forward, parameters/named_parameters, train/eval, zero_grad,
state_dict/load_state_dict and offers `all_reduce_gradients()` and
`wait_for_gradient_sync()`. Use sync_grads=False for master-gradient AMP-DDP.
The launcher is `python -m orthon.distributed.run`; configuration and current
NCCL hardware/dependency limits are in [distributed runtime](distributed.md).

## Private implementation surfaces

`orthon._C`, `orthon._diagnostics`, `orthon._serialization`, `_ProcessGroup`,
`_Reducer`, private Tensor hooks/copy helpers and prepared-state services are
internal. They are not public application APIs. Architecture and benchmark
explanations may discuss them without adding them to the public inventory.

## Public-name manifest

The architecture test compares this explicit scope with real package exports.
Magic methods, enum members, incidental imported submodules and underscore names
are excluded. Tensor/Module methods are explained above rather than exported as
module-level names. Conditional CUDA diagnostic bindings remain private.

```json
{
  "orthon": [
    "DType",
    "Device",
    "DeviceType",
    "Tensor",
    "add",
    "amp",
    "argmax",
    "cast",
    "contiguous",
    "conv2d",
    "cross_entropy",
    "cuda_enabled",
    "cuda_synchronize",
    "div",
    "dropout",
    "embedding",
    "empty",
    "exp",
    "fill_",
    "flatten",
    "from_numpy",
    "full",
    "fused_linear_relu",
    "gelu",
    "is_grad_enabled",
    "isfinite_all",
    "layer_norm",
    "leaky_relu",
    "load_checkpoint",
    "load_state_dict",
    "log_softmax",
    "manual_seed",
    "matmul",
    "max",
    "maxpool2d",
    "mean",
    "mul",
    "nccl_enabled",
    "neg",
    "nll_loss",
    "nn",
    "no_grad",
    "ones",
    "optim",
    "reciprocal",
    "relu",
    "reshape",
    "save_checkpoint",
    "save_state_dict",
    "scaled_dot_product_attention",
    "set_grad_enabled",
    "sigmoid",
    "softmax",
    "sqrt",
    "sub",
    "sum",
    "tanh",
    "to",
    "training",
    "transpose",
    "zeros"
  ],
  "orthon.nn": [
    "BatchNorm1d",
    "BatchNorm2d",
    "Conv2d",
    "Dropout",
    "Embedding",
    "Flatten",
    "FusedLinearReLU",
    "GELU",
    "LayerNorm",
    "LeakyReLU",
    "Linear",
    "MaxPool2d",
    "Module",
    "MultiHeadSelfAttention",
    "NLLLoss",
    "Parameter",
    "ReLU",
    "SelfAttention",
    "Sequential",
    "Sigmoid",
    "Tanh",
    "TransformerEncoderBlock"
  ],
  "orthon.optim": [
    "Adam",
    "SGD"
  ],
  "orthon.amp": [
    "GradScaler",
    "MixedPrecisionOptimizer"
  ],
  "orthon.distributed": [
    "Backend",
    "DDP",
    "DistributedDataParallel",
    "Work",
    "all_reduce",
    "barrier",
    "broadcast",
    "destroy_process_group",
    "get_rank",
    "get_world_size",
    "init_process_group",
    "is_initialized",
    "rank_config_from_env",
    "register_backend",
    "spawn"
  ],
  "orthon.training": [
    "classification_accuracy",
    "eval_step",
    "train_step",
    "weighted_metric_average"
  ]
}
```
