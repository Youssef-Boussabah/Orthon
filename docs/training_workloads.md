# Training Workloads

A practical guide to `examples/` and `benchmarks/`: what each workload
demonstrates, what it needs, and how to run it. The training APIs these
scripts exercise are documented in [training_stack.md](training_stack.md);
benchmark numbers recorded from these scripts live in
[../profiles/RESULTS.md](../profiles/RESULTS.md).

## Conventions

- Every script runs from the repository root against an installed or
  PYTHONPATH-visible `orthon`.
- Metrics are written as JSON under `examples/results/` — local artifacts,
  gitignored, never committed.
- Long workloads are **capped by default** (`--max-train-batches`,
  `--max-test-batches`) so a first run is bounded; pass `0` for full epochs.
- `--device cpu|cuda` selects the device; CUDA scripts check
  `orthon.cuda_enabled()` and fail with a clear message on CPU-only builds.
- Dataset-driven scripts (MNIST, CIFAR-10) need `torch` and `torchvision`
  **for data loading only**; synthetic-data workloads need neither.
  `--synthetic-data` (CIFAR) substitutes generated data for smoke runs with no
  download.

| Script | Demonstrates | Device | Data |
| --- | --- | --- | --- |
| `examples/mnist_mlp.py` | End-to-end MLP training with `train_step`/`eval_step` | CPU | MNIST (torchvision) |
| `examples/cifar10_cnn.py` | Conv2d stack, two model sizes, augmentation, LR schedules, overfit diagnostics, profiling | CPU/CUDA | CIFAR-10 or `--synthetic-data` |
| `examples/tiny_transformer_classifier.py` | Transformer encoder end to end on a synthetic task | CPU/CUDA | synthetic |
| `examples/tiny_causal_lm.py` | Causal masking, autoregressive greedy generation | CPU/CUDA | synthetic |
| `examples/cuda_amp_mlp.py` | FP16 master-weight + GradScaler loop | CUDA | synthetic |
| `examples/synthetic_mlp_checkpoint.py` | Checkpoint save/resume equivalence | CPU | synthetic |
| `examples/ddp_local_demo.py` | DDP on the local multiprocessing backend | CPU | synthetic |
| `examples/nn_api_demo.py` | The `nn` API surface in one file | CPU/CUDA (auto) | synthetic |
| `examples/tiny_text_classifier.py` | Embedding + attention on a tiny task | CPU/CUDA | synthetic |
| `*_torch.py` variants | The same workload in PyTorch, as the speed/accuracy reference | — | same |

## MNIST MLP

`Linear(784, hidden) -> ReLU -> Linear(hidden, 10)`, flattened 28×28 images,
cross-entropy loss. The Orthon script and the PyTorch baseline share batch
size, hidden size, optimizer, learning rate, epoch count and batch caps, so
their JSON outputs are directly comparable.

```bash
python examples/mnist_mlp.py --epochs 1
python examples/mnist_mlp_torch.py --epochs 1
python examples/compare_mnist_results.py                 # side-by-side table
python examples/compare_mnist_results.py --csv examples/results/mnist_comparison.csv
```

Defaults cap at 100 train / 50 test batches. A successful run shows Orthon
loss decreasing and accuracy above chance, tracking the PyTorch baseline
directionally; PyTorch is expected to be faster.

## CIFAR-10 CNN

Exercises `Conv2d -> ReLU -> MaxPool2d -> Flatten -> Linear -> cross_entropy`
in a real convolutional workload. Two models:

- `--model small` (~34k parameters) — fast diagnostic CNN for parity, smoke
  tests and fixed-batch debugging.
- `--model minivgg` (~419k parameters) — VGG-style 32/64/128-channel blocks
  for stronger accuracy experiments.

```bash
python examples/cifar10_cnn.py --epochs 1                          # capped default
python examples/cifar10_cnn.py --synthetic-data --epochs 1 \
    --max-train-batches 2 --max-test-batches 2                     # no dataset needed
python examples/cifar10_cnn_torch.py --epochs 1                    # PyTorch baseline
```

Accuracy-path controls for MiniVGG: `--augment` (RandomCrop + horizontal
flip), `--weight-decay` (decoupled; biases skipped unless `--decay-bias`),
`--lr-schedule step|cosine` with `--lr-step-epochs`/`--lr-gamma`/`--min-lr`
(built on the optimizers' `get_lr()`/`set_lr()`).

Recommended full MiniVGG run:

```bash
python examples/cifar10_cnn.py --model minivgg --epochs 30 --batch-size 128 \
    --device cuda --optimizer adam --lr 0.001 --adam-eps 0.00000001 \
    --weight-decay 0.0005 --augment --lr-schedule step \
    --lr-step-epochs 15,25 --lr-gamma 0.1 \
    --max-train-batches 0 --max-test-batches 0
```

The recorded local result for this exact command is **84.11% best test
accuracy** (epoch 27) at roughly 24–29s per epoch on an RTX 4070 Laptop GPU —
a local result for this implementation and recipe, not a
state-of-the-art claim. Details and the accompanying profile breakdown are in
[../profiles/RESULTS.md](../profiles/RESULTS.md#minivgg-cifar-10).

### Overfit diagnostics

Before chasing accuracy, verify the training loop can memorize:
`--overfit-small --overfit-batches 2` collects a fixed set of raw batches once
at startup and trains/evaluates on exactly those, in order (augmentation is
disabled automatically). `--compare-cpu-cuda-step` runs one forward/backward
on the same fixed mini-batch on both devices for parity checking. If CPU
overfits and CUDA does not, suspect CUDA Conv2d backward or optimizer state;
if neither does, suspect architecture, initialization scale, hyperparameters
or the data loop. `--debug-logits` prints first-batch logit statistics; the
script warns on NaN/Inf losses and implausibly large initial loss.

### Profiling hooks

`--profile` records per-epoch section timing (`data_loading`,
`input_conversion`, `forward`, `loss`, `backward`, `optimizer`, `metrics`,
`evaluation`) into the metrics JSON; add `--profile-sync` to bracket GPU
sections with `orthon.cuda_synchronize()` — without synchronization, queued
CUDA work is misattributed to later Python sections. Setting
`ORTHON_PROFILE_CONV2D=1` additionally aggregates Conv2d backward time by
shape and by grad-input/grad-weight/grad-bias component
(`orthon.cuda_conv2d_profile_reset()` / `_summary()`).

### CUDA Conv2d kernel selectors

CUDA Conv2d backward dispatches to specialized kernels for the common 3×3
stride-1 padding-1 shape class (`optimized_v2` grad-input, `grouped3x3`
grad-weight) and generic optimized kernels otherwise; grad-bias is fused into
the grad-weight launch for small channel counts. Environment variables
override the choice, which the Conv2d benchmark uses for comparisons:

```powershell
$env:ORTHON_CUDA_CONV2D_FORWARD="naive"                    # or default tiled
$env:ORTHON_CUDA_CONV2D_GRAD_INPUT="naive"                 # naive|optimized|optimized_v2
$env:ORTHON_CUDA_CONV2D_GRAD_WEIGHT="grouped3x3"           # naive|optimized|grouped3x3|optimized_v2
$env:ORTHON_CUDA_CONV2D_BIAS_GRAD="fused"                  # naive|optimized|separate|fused
```

`ORTHON_DEBUG_DEVICE=1` logs explicit CUDA→CPU host materializations
(`item()`, `tolist()`) — useful for confirming the training path stays
device-resident; metric reporting intentionally materializes scalars.

## Tiny Transformer sequence classifier

Validates `nn.TransformerEncoderBlock` end to end — token + positional
embeddings, pre-norm encoder block, mean pooling, classifier head — on a
balanced synthetic task (detect a special token in an 8-token sequence over a
32-word vocabulary).

```bash
python examples/tiny_transformer_classifier.py --device cpu --epochs 20
python examples/tiny_transformer_classifier.py --device cuda --epochs 20 \
    --batch-size 64 --embed-dim 32 --num-heads 4 --lr 0.001
```

Recorded local CUDA runs reach 100% test accuracy by epoch 2 at ~0.13–0.15s
per epoch, which validates autograd through attention, LayerNorm, GELU and
the residual connections. Implementation notes: `nn.Linear` is 2-D, so the
block flattens `[B, T, E] -> [B*T, E]` around its linear layers;
`MultiHeadSelfAttention` uses `contiguous()` after transpose to feed reshape
without breaking the graph; the whole path is composition-based (no fused
kernels).

## Tiny causal language model

`examples/tiny_causal_lm.py` trains a causal next-token predictor on
synthetic arithmetic progressions (predict `(token + d) % vocab` for a hidden
step `d`), then generates autoregressively.

```bash
python examples/tiny_causal_lm.py --device cuda --epochs 30 \
    --batch-size 64 --embed-dim 32 --num-heads 4 --num-layers 1 --lr 0.001
```

What it demonstrates beyond the classifier:

- `make_causal_mask(seq_len, device)` builds a `[1, T, T]` additive mask
  (0 for allowed positions, −1e9 for future ones) that broadcasts over the
  `[B*H, T, T]` attention scores.
- Greedy generation stays device-resident: token accuracy uses
  `orthon.argmax(log_probs.reshape([B*T, V]), dim=-1)` on device and moves only
  the Int64 index tensor to CPU; each generated token comes from
  `orthon.argmax(log_probs[0][T-1]).item()` — one device-side scalar, no host
  round-trip for the logit tensor. Generation runs under `no_grad` with
  `model.eval()`.
- No sampling, temperature, top-k or beam search — greedy decoding only.

Recorded local CUDA runs reach ~96% token accuracy by epoch 30 with exact
generation (including the modular wrap) from epoch 5; the residual token error
is the inherent ambiguity of the first token, from which the step `d` cannot
be inferred. `tests/python/test_tiny_causal_lm.py` covers the mask, shapes,
full backward and generation.

## Mixed-precision smoke

```bash
python examples/cuda_amp_mlp.py --steps 5 --device cuda
```

Runs the FP16 master-weight + `GradScaler` loop from
[training_stack.md](training_stack.md#mixed-precision) on a synthetic MLP.
The AMP-vs-FP32 timing comparison is `benchmarks/bench_amp_vs_fp32.py`.

## Checkpoint resume smoke

```bash
python examples/synthetic_mlp_checkpoint.py --steps 5 --resume-smoke
```

Trains a synthetic MLP, checkpoints with
`orthon.save_checkpoint(path, model=..., optimizer=..., metadata=...)`,
restores into fresh objects and verifies resumed training matches. Format
guarantees: [checkpoint_format.md](checkpoint_format.md).

## Distributed demo

```bash
python examples/ddp_local_demo.py
python -m orthon.distributed.run --nproc_per_node 2 --backend nccl \
    benchmarks/benchmark_distributed_ddp.py --mode bucketed --bucket-size-mb 25
```

The local demo runs DDP over the multiprocessing backend on CPU; the NCCL
benchmark needs a CUDA+NCCL build and multiple GPUs. The full contract and
its validation status are in [ddp_cloud_validation.md](ddp_cloud_validation.md).

## Benchmarks

Focused scripts, each measuring one thing, all writing to
`examples/results/`:

```bash
python examples/benchmark_training.py --framework orthon --device cuda --preset small
python examples/benchmark_ops.py                     # Conv2d/matmul/cross-entropy latency
python examples/benchmark_conv2d.py                  # kernel-selector comparison vs PyTorch
python examples/benchmark_fusion.py                  # fused_linear_relu vs separate ops
python examples/benchmark_no_grad.py --device cuda   # graph-construction overhead
python examples/benchmark_dropout.py
python benchmarks/benchmark_memory_pool.py           # allocator reuse + scratch stats
python benchmarks/bench_matmul.py --device cuda
python benchmarks/bench_elementwise.py
python benchmarks/bench_reductions.py
python benchmarks/bench_softmax_cross_entropy.py --device cuda
python benchmarks/bench_amp_vs_fp32.py --device cuda --preset medium
python benchmarks/bench_tiny_mlp_training.py --steps 5 --device cuda
python benchmarks/bench_checkpoint_overhead.py
```

`benchmark_training.py` takes `--preset smoke|small|medium` (smoke validates
plumbing, medium is the first throughput-meaningful scale) and runs both an
MNIST-shaped MLP and a CIFAR-shaped CNN for `--framework orthon|torch` on
either device. Aggregate reports:

```bash
python examples/report_benchmarks.py --markdown examples/results/benchmark_report.md
python examples/report_training_results.py --markdown examples/results/training_report.md
```

### Reading benchmark output honestly

- These are narrow, hardware-specific microbenchmarks. PyTorch rows are
  reference measurements from the same script, shape, warmup and
  synchronization policy — not a general performance comparison.
- Tiny capped CUDA runs are dominated by per-launch overhead, Python dispatch
  and host↔device transfer; Orthon can look competitive there because the
  measured path has less machinery, while PyTorch is expected to win at scale
  with larger batches and production-tuned kernels.
- Operator backward timings include graph/tensor setup where a fresh graph is
  required per backward call; training benchmarks include Python-side data
  conversion.
- Do not read tiny-run accuracy numbers as model-quality claims.

Recorded results, with commands and system details, are collected in
[../profiles/RESULTS.md](../profiles/RESULTS.md); the Nsight-based workflow
for kernel-level analysis is in [nsight_profiling.md](nsight_profiling.md).
