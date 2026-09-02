# Training Stack

The Python-facing model and training system: `nn.Module` and the layers,
optimizers, mixed precision, and how they interact with checkpointing. This
layer is deliberately pure Python over public tensor operations — it inherits
device support, dtype behaviour and autograd from the native runtime
([architecture.md](architecture.md)) instead of re-implementing any of them,
which is why every layer works on CPU and CUDA without device-specific code.

Related: [autograd.md](autograd.md) for what `backward()` does,
[checkpoint_format.md](checkpoint_format.md) for the on-disk format,
[training_workloads.md](training_workloads.md) for runnable end-to-end
examples.

## Modules and parameters

`nn.Module` (`python/orthon/nn/module.py`) mirrors the PyTorch idiom: assign a
`Parameter` or a child `Module` as an attribute and it is registered
automatically; write `forward()`; call the module.

```python
import orthon as ortho
from orthon.nn import Linear, Module


class MLP(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(64, 32)
        self.fc2 = Linear(32, 10)

    def forward(self, x):
        return self.fc2(ortho.relu(self.fc1(x)))
```

- **`Parameter`** wraps a tensor and sets `requires_grad=True`. Registration
  happens in `Module.__setattr__`. `parameters()` walks the module tree
  depth-first and deduplicates shared tensors by object identity — a tensor
  registered under two names is yielded once. `named_parameters()` recurses
  over the registered names without that deduplication, so a shared tensor
  appears once per name it is registered under.
- **Buffers** — non-trainable state such as BatchNorm's running statistics —
  are registered with `register_buffer(name, tensor, persistent=True)`. They
  move with `.to()`, cast with `.cast()` (floating buffers only, so a
  model-wide `half()` cannot corrupt an integer index table), and persistent
  ones appear in `state_dict()`. A buffer may be registered as `None` to
  declare the slot before materializing it.
- **`train()` / `eval()`** toggle the `training` flag recursively; `Dropout`
  and BatchNorm read it.

### `state_dict` and loading

`state_dict()` returns `{name: tensor}` for parameters and persistent
buffers, with dotted hierarchical names (`"fc1.weight"`). The values are
**live references**, not copies — which is what makes
`model.load_state_dict(model.state_dict())` and in-place optimizer updates
coherent. The file APIs snapshot instead; see
[checkpoint_format.md](checkpoint_format.md#snapshot-vs-live-reference).

`load_state_dict(state, strict=True)` validates everything first (key sets,
shapes, types), prepares device- and dtype-reconciled copies, and only then
commits. Parameters are copied **into**, never replaced, so optimizer
references, `requires_grad`, `.grad` and post-accumulate hooks all survive a
load.

### Device movement and dtype casts

`Module.to(device)`, `Module.cast(dtype)`, and the `half()` / `bfloat16()`
shorthands walk the tree and transform parameters and buffers in place,
returning `self`. Public `to()` and floating `cast()` are differentiable
tensor ops, so these model-state transformations run under explicit `no_grad`
regions and re-assert `requires_grad` — every moved parameter stays a detached
leaf rather than picking up a graph edge.

## Layers

All composition-based — no layer owns a private kernel; each is a `forward()`
over public ops:

| Group | Modules |
| --- | --- |
| Core | `Linear`, `Sequential`, `Flatten`, `FusedLinearReLU` |
| Activations | `ReLU`, `Sigmoid`, `Tanh`, `LeakyReLU`, `GELU` |
| Convolutional | `Conv2d`, `MaxPool2d` |
| Normalization | `LayerNorm`, `BatchNorm1d`, `BatchNorm2d` |
| Regularization | `Dropout` (inverted, native CPU/CUDA masks) |
| Embedding | `Embedding` (with `padding_idx` gradient skipping) |
| Attention | `SelfAttention`, `MultiHeadSelfAttention`, `TransformerEncoderBlock` |
| Loss | `NLLLoss` (plus functional `ortho.cross_entropy`, `ortho.nll_loss`) |

Notes that matter when composing them:

- `Linear` takes 2-D input `[N, in_features]`; the Transformer block flattens
  `[B, T, E] → [B*T, E]` around its linear layers.
- `TransformerEncoderBlock` is pre-norm
  (`x + dropout(MHA(LN(x)))`, then the GELU MLP), built from LayerNorm,
  multi-head attention, Dropout and Linear — no fused attention or LayerNorm
  kernels.
- `ortho.cross_entropy(logits, targets)` takes targets as a Python sequence of
  class indices; `ortho.nll_loss(log_probs, targets, reduction=...)` accepts
  an index tensor or sequence, with `mean`/`sum`/`none` reductions.
- BatchNorm is composition-based over Tensor reductions and elementwise
  operations, so its statistics stay on the input's device (CPU or CUDA).
  Running statistics are detached through explicit no-grad same-device Tensor
  snapshots — no host round-trip. There is no dedicated fused BatchNorm CUDA
  kernel (see [cuda_coverage.md](cuda_coverage.md)).

## Initialization and RNG

Layers draw initial weights (e.g. `Linear`'s Kaiming-uniform init) from **one
framework-local RNG stream** (`python/orthon/_random.py`) — deliberately not
Python's global `random` state, so unrelated user code shuffling a dataset
cannot silently shift model initialization, and seeding Orthon cannot disturb
the user's own sampling.

```python
ortho.manual_seed(7)   # opt-in determinism for module initialization
```

`manual_seed` resets that one stream and nothing else — not NumPy, not
PyTorch, not CUDA device state, not existing tensors. A distributed user who
wants rank-specific initialization calls `manual_seed(base + rank)`. Dropout
additionally supports per-module deterministic seeding; there is no global
stream-aware RNG system beyond this.

## Optimizers

`optim.SGD` and `optim.Adam` hold a list of parameter tensors and step them
inside `no_grad` using in-place ops, so a step never enters the autograd
graph.

- **`SGD(params, lr, weight_decay=0.0, decay_bias=False)`** — plain
  `p -= lr * grad` (no momentum), with decoupled weight decay that skips 1-D
  (bias-shaped) parameters unless `decay_bias=True`.
- **`Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
  decay_bias=False)`** — standard bias-corrected moments; moment tensors are
  created lazily on the parameter's own device at the first step, and the
  weight decay is decoupled (AdamW-style).

Both expose `get_lr()` / `set_lr()` for epoch-boundary schedules, and
`state_dict()` / `load_state_dict()` for resume. Optimizer state restored from
a checkpoint lands on the parameters' current devices.

`orthon.training` adds three small loop helpers used by the examples —
`train_step`, `eval_step`, `weighted_metric_average` — for supervised
classification; they are conveniences, not a trainer framework.

## Mixed precision

`orthon.amp` implements explicit-cast mixed precision: FP16 or BF16 forward
and backward, FP32 optimizer mathematics. There is **no autocast** — the
model is converted once with `model.half()` / `model.bfloat16()`, and inputs
are cast at the boundary.

**`MixedPrecisionOptimizer(model, OptimizerCls, **kwargs)`** creates an FP32
**master copy** of every low-precision parameter (a detached leaf, snapshotted
under `no_grad`) and hands the masters to the inner optimizer. Each `step()`
casts low-precision gradients up to FP32, steps the masters, and casts the
result back into the model. FP32 parameters skip the machinery entirely — the
master *is* the model parameter, so an FP32 model pays no copy overhead.

**`GradScaler`** adds dynamic loss scaling for FP16, whose narrow exponent
range lets small gradients underflow to zero:

- `scale(loss)` multiplies by the current scale — a Python-scalar multiply, so
  no loss-shaped constant tensor is built;
- `unscale_(opt)` divides FP32 master gradients by the scale and records
  whether any are non-finite (idempotent, so explicit unscale-then-clip
  works);
- `step(opt)` unscales if that has not happened yet, synchronizes the
  finite-gradient decision (across ranks under DDP), and then either skips the
  optimizer update and backs the scale off (non-finite gradients found) or
  steps the masters and grows the scale after `growth_interval` consecutive
  clean steps — scale adaptation lives in `step()`, not `update()`;
- `update()` clears the per-step unscale/finite state so the next iteration
  starts fresh.

BF16 has float32's exponent range, so the BF16 path is
`GradScaler(enabled=False)` — same loop shape, scaling off. Both the scaler
and the mixed-precision optimizer expose `state_dict()`/`load_state_dict()`,
so scale state and FP32 masters survive checkpoint/resume. Under DDP, ranks
synchronize their non-finite-gradient decisions so all ranks skip (or step)
together — see [ddp_cloud_validation.md](ddp_cloud_validation.md).

The dtype/kernel substrate this rides on (FP16/BF16 CUDA kernels, cuBLAS
Tensor Core dispatch, CUDA cast coverage) is described in
[cuda_runtime.md](cuda_runtime.md#cuda-dtype-support).

## A complete training loop

Synthetic two-class problem, CPU, with checkpoint/resume — every API here is
the real one:

```python
import numpy as np
import orthon as ortho
from orthon.nn import Linear, ReLU, Sequential
from orthon.optim import Adam

ortho.manual_seed(0)

model = Sequential(Linear(16, 32), ReLU(), Linear(32, 2))
opt = Adam(model.parameters(), lr=1e-2)

rng = np.random.default_rng(0)
features = rng.standard_normal((64, 16)).astype(np.float32)
labels = [int(row[0] > 0) for row in features]        # class = sign of feature 0
x = ortho.from_numpy(features)

for step in range(20):
    model.train()
    opt.zero_grad()
    loss = ortho.cross_entropy(model(x), labels)
    loss.backward()
    opt.step()

ortho.save_checkpoint(
    "mlp.ckpt", model=model, optimizer=opt, metadata={"step": 20},
)

# Later, or elsewhere: rebuild the architecture, then restore everything.
model2 = Sequential(Linear(16, 32), ReLU(), Linear(32, 2))
opt2 = Adam(model2.parameters(), lr=1e-2)
meta = ortho.load_checkpoint("mlp.ckpt", model=model2, optimizer=opt2)
assert meta["step"] == 20

model2.eval()
with ortho.no_grad():
    logits = model2(x)
```

Moving this loop to CUDA is `model.to("cuda")` plus `x.to("cuda")`; making it
mixed-precision is `model.half()` plus the `MixedPrecisionOptimizer` /
`GradScaler` pair (the runnable CUDA AMP workload is
[training_workloads.md](training_workloads.md#mixed-precision-smoke)).

## Checkpoint interaction

`save_checkpoint(path, model=..., optimizer=..., metadata=...)` snapshots
model state, optimizer state (including AMP masters and scaler state when you
pass their owners' state through) and a metadata dict into one
`ml-runtime-checkpoint` v1 archive; `load_checkpoint` validates the entire
file before mutating anything and returns the metadata. Devices are a *load*
policy: by default state loads onto the model's current devices, and
`map_location="cuda:1"` overrides. The format's guarantees — no pickle,
bit-exact round-trips, atomic saves — are specified in
[checkpoint_format.md](checkpoint_format.md).

## Boundaries

- No autocast, no learning-rate scheduler objects (schedules are a few lines
  over `set_lr`), no parameter groups, no gradient clipping helper (clip
  between `unscale_` and `step` by hand), no momentum SGD.
- No trainer/loop abstraction; the examples own their loops.
- DDP wraps this stack from the outside via post-accumulate gradient hooks —
  the layers and optimizers are unaware of it.
