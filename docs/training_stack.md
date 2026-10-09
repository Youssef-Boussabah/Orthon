# Training stack

Orthon's Python modules, optimizers and explicit AMP wrapper retain native Tensor
owners. The [API reference](api.md) lists the supported signatures; the
[coverage matrix](cuda_coverage.md) bounds operator/dtype combinations.

## Module registration and stable parameters

`python/orthon/nn/module.py` uses one private `Module._set_member` protocol for
parameter, child module, buffer and ordinary attribute registration. A name has
one category; reassignment removes stale registration in the other categories.
`Parameter(data)` marks a Tensor trainable; assigning it registers its data Tensor.
These private registration/preparation methods are not application extension APIs.

`parameters()` deduplicates by Python Tensor owner and includes frozen registered
parameters. `named_parameters()` preserves registered paths, including ties.
Two paths pointing to the same Python owner are tied parameters. Two distinct
owners sharing one Buffer remain distinct model/optimizer bindings.

Supported `to`, `cast`, `half` and `bfloat16` prepare transformed representations
for the whole module tree before publication. Parameter Python objects survive,
so optimizers constructed earlier still see their current device/dtype. Rebinding
native representation publishes leaf state and clears gradients
when the representation actually changes. Native autograd metadata belongs to
that representation; transform the model before attaching DDP hooks. A same-representation cast can preserve
existing grad state. `to()` snapshots even on the same device and therefore
normally changes the representation. Buffer-only owners are replaced; buffer ties
and parameter-buffer ties are preserved during the supported transform protocol.
Dtype transforms touch floating buffers; integer/bool buffers retain their dtype.

Replacing a registered parameter explicitly with a new Parameter is a different
operation: an existing optimizer still owns its original bound Tensor. Construct
an optimizer for the intended new owners.

## Persistent buffers and state

`register_buffer(name, tensor, persistent=True)` accepts Tensor or None. None
remains a registered slot but is absent from tensor enumeration/state. Nonpersistent
buffers transform with the model but are omitted from state_dict. BatchNorm's
running mean/variance are persistent buffers. `train()`/`eval()` recursively
change mode; changing mode does not by itself disable autograd.

`state_dict()` exposes live Tensor references in registration order. File saving
snapshots them. Direct Module loading and file restore share a private
READ/VALIDATE/PREPARE/COMMIT protocol: all incoming values, shapes, keys and
conversions are prepared before publishing. Conflicting values for tied parameter
paths reject before commit. Default parameter load copies values without replacing
owners/metadata; buffer slots receive independent prepared values. Explicit device
remapping prepares parameter representations and nonpersistent buffer moves too.
See [checkpoint interaction](#checkpoint-interaction).

## Optimizer ownership

`python/orthon/optim/_binding.py` captures strong Python parameter-owner bindings.
Duplicate owner entries reject; distinct storage aliases remain separate bindings.
A binding consults its owner's current representation at step time.

`SGD(params, lr, weight_decay=0.0, decay_bias=False)` stores configuration only.
It updates parameters with gradients under no_grad and supports decoupled weight
decay, skipping one-dimensional bias-like values unless requested. There is no
momentum or parameter-group API.

`Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
decay_bias=False)` holds a paired lazy moment record per binding and one global
step counter. Both moments initialize together; supported placement/dtype changes
prepare and replace the pair together. Bare Adam supports CPU F32/F64 and CUDA
F32. Public state_dict serializes `m`/`v` lists, but those lists are not separate
runtime parameter owners. Prepared loading validates count, shape, moment pairing
and current/intended future placement before publication. Optimizer-only loading
cannot relocate bound parameters through conflicting map_location.

Both optimizers provide `zero_grad`, `step`, `get_lr`, `set_lr`, `state_dict` and
`load_state_dict`. Gradients accumulate until cleared. Normal step execution is
not a whole-optimizer transaction for arbitrary runtime exceptions.

## Mixed precision

`python/orthon/amp/optimizer.py` stores one immutable state entry per model owner:
current model dtype, exact device, FP32 master and parameter owner. FP32 model
owners are their own masters. F16/BF16 owners use distinct detached FP32 masters.
The inner optimizer binds to these masters. Supported Module transforms reconcile
lazily; prepared state loading builds all needed masters and inner state before
publication. Explicit parameter replacement does not retarget existing bindings.
Master/model synchronization requires matching shape/exact device and contiguous
representations.

There is no autocast. Low-precision compute is principally CUDA; cast the model
and inputs explicitly. GradScaler supports scale, idempotent unscale/finiteness
checking, finite step, nonfinite skip/backoff and growth. Adaptation occurs in
`step`; `update()` clears cycle bookkeeping. One enabled optimizer participates
per cycle. Disabled mode calls the wrapper step and is suitable for BF16.
A scaler state dictionary can be saved explicitly in checkpoint metadata, but
restore does not apply it automatically.

## AMP with DDP

Use `DDP(model, sync_grads=False)` before constructing MixedPrecisionOptimizer,
then use GradScaler with that wrapper. Direct DDP low-precision gradient averaging
is unsupported; the wrapper prepares and synchronizes FP32 master gradients.
Ranks agree on nonfinite decisions for enabled scaling. Disabled scaling/BF16
still synchronizes master gradients when a multi-rank group is active. Ordinary
DDP captures its ProcessGroup at construction; AMP also retains its synchronization
group and checks its lifetime. Follow the initialized group's lifetime through
training, and destroy it after work completes. See [distributed](distributed.md).

## CPU training and checkpoint sample

This complete example uses only synthetic tensors and a temporary local archive:

```python
import tempfile
from pathlib import Path
import orthon as o

model = o.nn.Sequential(o.nn.Linear(4, 8), o.nn.ReLU(), o.nn.Linear(8, 2))
optimizer = o.optim.Adam(model.parameters(), lr=0.001)
x = o.full([3, 4], 0.2)
metrics = o.training.train_step(model, x, [0, 1, 0], o.cross_entropy, optimizer)
print(metrics["loss"])
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "training.ckpt"
    o.save_checkpoint(path, model=model, optimizer=optimizer, metadata={"step": 1})
    resumed = o.nn.Sequential(o.nn.Linear(4, 8), o.nn.ReLU(), o.nn.Linear(8, 2))
    resumed_optimizer = o.optim.Adam(resumed.parameters())
    metadata = o.load_checkpoint(path, model=resumed, optimizer=resumed_optimizer)
    print(metadata["step"])
```

`training.train_step` trains, clears gradients, computes loss/backward and steps.
`eval_step` sets evaluation mode but does not internally enter no_grad; callers
can wrap evaluation explicitly. `classification_accuracy` expects `[N,C]` logits
and integer targets; `weighted_metric_average` weights loss/accuracy by example
count and rejects zero total examples.

## Checkpoint interaction

The file facade delegates to private SerializationService and RestoreTransaction.
The whole present known archive is decoded/validated, requested recipients are
prepared, then model and optimizer commit in that order. Preparation receives
intended future parameter placement, enabling supported combined device remapping
without breaking optimizer owner references. Synchronous commit errors attempt
reverse rollback, including the failing participant. This is not recovery from
arbitrary asynchronous CUDA hardware faults.

The archive stores model, optimizer and metadata only. Training mode, gradients,
RNG, GradScaler, ProcessGroup and Reducer state are not automatically restored.
File payloads do not reconstruct a source alias graph; destination ties are
preserved. See [checkpoint format](checkpoint_format.md).

## RNG and application scope

`manual_seed(seed)` resets the framework-local Python initialization/dropout seed
stream in `python/orthon/_random.py`; it does not reset NumPy, PyTorch, CUDA global
state or provide a generator object API. LayerNorm, BatchNorm and attention are
compositions of supported primitives. Linear is 2-D; attention reshapes its
projections and uses equal-batch 3-D matmul. The current runtime provides neither
higher-order autograd nor a general framework parity contract. Runnable workload
boundaries are in [training workloads](training_workloads.md).
