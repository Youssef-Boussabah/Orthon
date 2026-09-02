# Orthon vs. PyTorch — Full Comparison

The complete head-to-head measurement of Orthon 0.1.0 against PyTorch eager on
one RTX 4070 Laptop GPU: every workload measured, wins and losses alike, with
the methodology stated precisely enough to be attacked.

**What this document does not say.** It does not say Orthon is faster than
PyTorch. Orthon loses most of the matrix below — decisively on FP32 matmul and
Conv2d, and on cross entropy and Linear+ReLU outside a specific shape band.
The question this document answers is narrower and more interesting: *on which
concrete workloads does the frozen Orthon 0.1.0 runtime actually outperform
PyTorch eager on the same GPU, and why?*

Every number here was produced by
[`benchmarks/compare_pytorch.py`](../benchmarks/compare_pytorch.py) in this
pass. Nothing was carried forward from an earlier table.

- [Environment](#environment)
- [Methodology](#methodology)
- [Numerical equivalence policy](#numerical-equivalence-policy)
- [Fairness notes per workload](#fairness-notes-per-workload)
- [Full result matrix](#full-result-matrix)
- [Stable Orthon wins](#stable-orthon-wins)
- [Ties and unreliable orderings](#ties-and-unreliable-orderings)
- [PyTorch wins](#pytorch-wins)
- [Reproducibility of the headline rows](#reproducibility-of-the-headline-rows)
- [Selected portfolio results](#selected-portfolio-results)
- [Limitations](#limitations)
- [Reproduction commands](#reproduction-commands)

## Environment

Captured automatically by the harness, not hard-coded:

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |
| Compute capability | `sm_89` |
| GPU memory | 8.0 GiB |
| GPU L2 cache | 32.0 MiB |
| SM count | 36 |
| NVIDIA driver | 596.08 |
| Platform | Windows 11 |
| Python | 3.12.6 |
| NumPy | 2.5.2 |
| PyTorch | 2.13.0+cu130 |
| PyTorch CUDA runtime | 13.0 |
| cuDNN | 92000 |
| Orthon | 0.1.0 |
| Orthon CUDA build | enabled |
| Orthon extension SM targets | `sm_89` (native cubin, no PTX JIT) |
| `torch.backends.cuda.matmul.allow_tf32` | `False` |
| `torch.backends.cudnn.allow_tf32` | `False` |
| `torch.get_float32_matmul_precision()` | `highest` |

**The Orthon build matters and is reported for a reason.** The extension was
built with `CMAKE_CUDA_ARCHITECTURES=89`, so every Orthon kernel runs as native
`sm_89` SASS. A default-architecture build of this repository on CUDA 13.2
targets `sm_75` and reaches an Ada GPU only through PTX JIT, which is a
different measurement. The harness prints the extension's real SM targets in
its environment block so a reader can tell which one they are looking at. The
exact build and install command is in
[Reproducing the RTX 4070 results](#reproducing-the-rtx-4070-results).

This is the **performance** machine. It is not the 2x A100-SXM4-80GB machine
that produced the final multi-GPU correctness evidence — no timing in this
repository comes from that machine
([testing.md](../docs/testing.md#cuda-and-multi-gpu-boundary),
[ddp_cloud_validation.md](../docs/ddp_cloud_validation.md)).

## Methodology

**PyTorch mode: eager.** Every row below is Orthon against PyTorch **eager**.
`torch.compile` is not measured anywhere in this document. The harness
supports it (`--torch-compile`, wired for Fused Linear + ReLU) and would report
it as a separate row, never merged into an eager row, but Triton has no
Windows wheel and the call fails on this machine:

```
(torch.compile)  unavailable: TritonMissing: Cannot find a working triton installation.
```

If a compiled row is ever added, compilation happens during warmup and is
excluded from the timed samples; that exclusion would be stated in the row.

**What is inside the clock.** Only the operation. Before any timing starts,
every input tensor already exists on the GPU in its final dtype and layout.
Module import, CUDA context creation, host-to-device transfer, input
generation, weight initialization and the numerical check all happen first and
are excluded.

**Batched wall-clock timing.** Each sample is:

```python
sync()
t0 = perf_counter()
for _ in range(inner):
    fn()
sync()
latency = (perf_counter() - t0) * 1000 / inner
```

Timing a batch of `inner` repetitions rather than a single call keeps
sub-millisecond kernels out of Python timer noise. `sync()` is each framework's
own synchronization call — `orthon.cuda_synchronize()` and
`torch.cuda.synchronize()`.

**`inner` is derived, and identical for both frameworks in a case.** The
harness estimates each framework's per-call cost, then picks one `inner` such
that the faster framework's sample contains at least `--target-ms` (default
40 ms) of GPU work, capped so the slower framework's sample stays under
`--max-sample-ms` (default 500 ms). Both frameworks then run that same `inner`.

**Warmup.** A one-second global GPU warmup runs before anything is measured, to
get the laptop GPU off its idle clock. Each case then warms each framework for
25 iterations (`--warmup`) before estimation, and estimation itself runs more
iterations before the first sample.

**Alternation.** 11 independent samples per framework (`--samples`). Rounds
alternate: Orthon first on even rounds, PyTorch first on odd rounds. Neither
framework systematically gets the warmer or the cooler GPU.

**Statistics.** Reported per row: median latency, interquartile range (IQR),
and speedup = `median(PyTorch) / median(Orthon)`. The fastest single run is
never reported. Two derived judgements:

- **Ordering reliability.** The speedup band `[p25(torch)/p75(orthon),
  p75(torch)/p25(orthon)]` is computed. If that band contains 1.0, the ordering
  is not established by the data and the row is labelled **ordering
  unreliable** — regardless of how the medians happen to fall.
- **Stability.** A row whose relative IQR exceeds 15% on either side is flagged
  `noisy`. A noisy row can still have a reliable ordering; it is excluded from
  portfolio selection anyway.

**Win threshold.** A row counts as an Orthon win only at **>= 10%** lower
median latency with a reliable ordering. Between 0.909x and 1.10x the row is
reported as **similar**, never as a victory.

## Numerical equivalence policy

Every case computes both frameworks' results and compares them **before** it is
timed. A row that fails is reported as `NUMERICAL CHECK FAILED`, carries no
performance claim, and makes the harness exit non-zero.

The metric is the maximum absolute difference scaled by the reference's own
maximum magnitude, with PyTorch eager as the reference. Tolerances are per
dtype and per operation:

| Workload | Tolerance | Reason |
| --- | --- | --- |
| Elementwise add | `1e-7` | bit-identical arithmetic expected |
| Softmax | `1e-6` | same stable formulation, different reduction order |
| Cross entropy (forward, and forward + backward) | `1e-5` | float32 loss and gradient |
| FP32 matmul | `1e-4` | float32 accumulation order differs |
| Fused Linear + ReLU | `1e-4` | float32 accumulation order differs |
| Conv2d | `3e-4` | `C*K*K` term accumulation order differs |
| Sum reduction | `1e-5` | tree vs sequential reduction order |
| BF16 matmul | `3e-2` | bfloat16 output rounding (~2^-8 relative) |

For the backward case both the loss value **and** `d(loss)/d(logits)` are
compared, on freshly built tensors so the timing tensors' accumulated
gradients cannot contaminate the check.

**All 36 cases passed.** Observed relative differences ranged from exactly
`0.00e+00` (cross entropy forward, elementwise add, BF16 matmul) to `2.26e-07`
(2048x4096 softmax) — every one of them orders of magnitude inside its
tolerance.

## Fairness notes per workload

**FP32 matmul.** TF32 is disabled on the PyTorch side
(`torch.backends.cuda.matmul.allow_tf32 = False`,
`torch.set_float32_matmul_precision("highest")`) because Orthon's float32 path
is a handwritten tiled kernel doing strict float32 FMA. Both sides therefore
compute strict float32. Enabling TF32 for PyTorch — `--tf32`, which relabels
the rows and loosens the tolerance to `1e-2` — makes PyTorch faster still; it
is a deliberately different, separately labelled comparison, and no TF32 row
appears in the matrix below. The `--tf32` flag also governs cuDNN, so Conv2d is
strict float32 on both sides too.

**BF16 matmul.** Inputs are rounded to bfloat16 by Orthon's own conversion
first; the resulting exactly-representable float32 values are then handed to
PyTorch and cast. Both frameworks therefore multiply **bit-identical** inputs,
and neither pays a conversion the other does not. Both produce bfloat16 output
with float32 accumulation (Orthon dispatches to
`cublas-bfloat16-fast-bf16-tensor-op`, reported in the row header). The
measured maximum difference is exactly zero at every shape, which is itself
evidence that the two are doing the same arithmetic.

**Cross entropy.** Same logits, same dense class-index targets, same mean
reduction, same float32 dtype. The two APIs differ in one respect that is
Orthon's, not the harness's: `orthon.cross_entropy` takes targets as a host
`list[int]` and uploads them on every call, along with a `cudaMalloc`/
`cudaFree` pair, while `F.cross_entropy` reads an int64 CUDA tensor placed
before timing. That cost stays inside Orthon's timed region because there is no
way to call the op without it. It handicaps Orthon, visibly so — see the
`8192x512` row, where 8192 host indices per call cost more than the entire
loss — and it means the wins below are wins *despite* it.

**Cross entropy forward + backward** is reported as its own benchmark and never
mixed with the forward-only rows. Both frameworks accumulate gradients into the
same tensors across iterations, symmetrically.

**Fused Linear + ReLU.** Orthon runs one fused CUDA kernel. PyTorch eager runs
`F.relu(F.linear(x, W, b))` — the fastest idiomatic eager spelling of a
Linear + ReLU layer, which reaches a cuBLAS(Lt) `addmm` with a fused bias
epilogue in two dispatches.

This baseline was chosen deliberately, and it replaced a weaker one. An
earlier revision of this harness timed PyTorch as the explicit composition
`torch.relu(x @ W + b)` — three dispatches — on the grounds that both sides
then held the same `K x N` weight. That argument does not survive scrutiny:
PyTorch's own `nn.Linear` does not run three dispatches, so measuring against
the three-dispatch spelling charged PyTorch for work it does not actually do,
and it inflated Orthon's margin at the shapes where the two frameworks are
close. The explicit composition is retained only as a numerical cross-check
in `check()`; it sets no speedup, no verdict, and no published claim.

Fairness of the layout is handled where it belongs, outside the timed region.
PyTorch's Linear weight is `(out_features, in_features)` — `N x K`, the
transpose of Orthon's `K x N` — so the transpose and its contiguous copy are
performed once at case construction. Both frameworks hold the same
mathematical weight, each in its own preferred physical layout, and neither
pays a conversion at call time.

Orthon's advantage remains confined to small shapes: the fused kernel
accumulates over `K` in a single thread with no shared-memory tiling, so once
`K` grows past ~512 cuBLAS wins by a wide margin, and the matrix below
publishes that.

**Conv2d.** Orthon's handwritten CUDA convolution against PyTorch eager, which
reaches cuDNN. Included precisely because it is the workload where Orthon
loses hardest.

## Full result matrix

Canonical run: `python benchmarks/compare_pytorch.py`, 11 samples per
framework, defaults otherwise. `Speedup` is `PyTorch / Orthon`, so above 1.00x
favours Orthon. `Max rel diff` is the numerical check result for that row.

### Cross entropy (forward)

float32, forward. Orthon takes host class indices and uploads them per call;
PyTorch reads a pre-placed int64 CUDA tensor.

| Shape | Orthon median ms | Orthon IQR | PyTorch median ms | PyTorch IQR | Speedup | Verdict | Max rel diff |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 512x1024 | 0.0817 | 0.0060 | 0.0375 | 0.0033 | 0.46x | PyTorch faster | 0.00e+00 |
| 1024x2048 | 0.1028 | 0.0059 | 0.0450 | 0.0022 | 0.44x | PyTorch faster | 0.00e+00 |
| 2048x4096 | 0.1752 | 0.0037 | 0.5363 | 0.0065 | **3.06x** | Orthon faster | 0.00e+00 |
| 4096x4096 | 0.5783 | 0.0182 | 1.0421 | 0.0202 | 1.80x | Orthon faster | 0.00e+00 |
| 8192x512 | 0.3801 | 0.0105 | 0.0935 | 0.0048 | 0.25x | PyTorch faster | 7.07e-08 |

The pattern is a class-count effect, not a size effect. Orthon's fused kernel
reads the logits once and writes `B` scalars; PyTorch eager materializes a
full `B x C` log-softmax and then gathers. That trade only pays when `C` is
large, and the per-call host-target upload — which grows with `B`, not `C` —
eats the rest. `8192x512` and `2048x4096` contain the same number of elements
and land on opposite sides of the result.

### Cross entropy (forward + backward)

float32, forward + backward: loss value and `d(loss)/d(logits)`.

| Shape | Orthon median ms | Orthon IQR | PyTorch median ms | PyTorch IQR | Speedup | Verdict | Max rel diff |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 512x1024 | 0.1847 | 0.0095 | 0.2544 | 0.0195 | 1.38x | Orthon faster | 5.96e-08 |
| 1024x2048 | 0.2782 | 0.0212 | 0.4001 | 0.0069 | 1.44x | Orthon faster | 5.96e-08 |
| 2048x4096 | 1.6369 | 0.0438 | 2.2773 | 0.0210 | 1.39x | Orthon faster | 5.96e-08 |
| 4096x4096 | 3.1429 | 0.0994 | 4.5233 | 0.0576 | **1.44x** | Orthon faster | 5.96e-08 |

### Softmax

float32, forward. Orthon fused row-wise CUDA kernel vs PyTorch eager softmax.

| Shape | Orthon median ms | Orthon IQR | PyTorch median ms | PyTorch IQR | Speedup | Verdict | Max rel diff |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 256x512 | 0.0144 | 0.0013 | 0.0179 | 0.0015 | 1.24x | Orthon faster | 7.09e-08 |
| 512x1024 | 0.0191 | 0.0019 | 0.0187 | 0.0014 | 0.98x | ordering unreliable | 9.42e-08 |
| 1024x2048 | 0.0421 | 0.0052 | 0.0368 | 0.0039 | 0.88x | ordering unreliable | 9.33e-08 |
| 2048x4096 | 0.5268 | 0.0324 | 0.5208 | 0.0209 | 0.99x | ordering unreliable | 2.26e-07 |

Softmax is a tie. The `256x512` row reads 1.24x here, but it is the smallest
shape in the family and every larger one lands at or below parity with an
unreliable ordering — 0.98x, 0.88x, 0.99x — so the family as a whole
establishes nothing. At the large shape both frameworks are within 1% of each
other and bandwidth-bound. The earlier hypothesis that Orthon's fused softmax
kernel is meaningfully faster than PyTorch's **does not survive this
harness**, and no softmax speed claim against PyTorch is made anywhere in this
repository.

### Fused Linear + ReLU

float32, forward. Orthon runs one fused CUDA kernel; PyTorch eager runs
`F.relu(F.linear(x, W, b))` on its native `N x K` weight, prepared before
timing.

| Shape | Orthon median ms | Orthon IQR | PyTorch median ms | PyTorch IQR | Speedup | Verdict | Max rel diff |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| M=128 K=256 N=128 | 0.0189 | 0.0025 | 0.0527 | 0.0054 | 2.79x | Orthon faster | 8.24e-07 |
| M=256 K=256 N=256 | 0.0535 | 0.0040 | 0.0454 | 0.0070 | 0.85x | PyTorch faster (noisy) | 7.22e-07 |
| M=256 K=512 N=256 | 0.1102 | 0.0067 | 0.0584 | 0.0059 | 0.53x | PyTorch faster | 1.32e-06 |
| M=512 K=512 N=512 | 0.4014 | 0.0149 | 0.0711 | 0.0065 | 0.18x | PyTorch faster | 1.41e-06 |
| M=512 K=1024 N=512 | 0.7874 | 0.0250 | 0.1372 | 0.0136 | 0.17x | PyTorch faster | 1.80e-06 |
| M=1024 K=1024 N=1024 | 3.0217 | 0.0716 | 0.4191 | 0.0196 | 0.14x | PyTorch faster | 1.56e-06 |

Against the idiomatic baseline the fusion advantage survives only at the
smallest shape. `M=128 K=256 N=128` wins at 2.79x; `M=256 K=256 N=256` — which
an earlier revision of this document published as a **1.68x Orthon win** — is
now a PyTorch win at 0.85x, and everything larger loses badly, up to 7.2x at
`M=1024 K=1024 N=1024` where Orthon's un-tiled inner loop over `K` meets
cuBLAS. That reversal is the whole point of correcting the baseline: the
earlier margin at this shape was an artefact of timing PyTorch as three
dispatches, not a property of the fused kernel.

The surviving `M=128 K=256 N=128` row is deliberately **not** used as a
headline either. It is the smallest shape in the matrix, both frameworks are
dominated there by per-call dispatch cost, and the ratio moved between 2.79x
and 3.24x across four runs with the PyTorch side noisy in three of them (see
[reproducibility](#reproducibility-of-the-headline-rows)).

### BF16 matmul

bfloat16 in, bfloat16 out, float32 accumulate. Orthon dispatch:
`cublas-bfloat16-fast-bf16-tensor-op`.

| Shape | Orthon median ms | Orthon IQR | PyTorch median ms | PyTorch IQR | Speedup | Verdict | Max rel diff |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 256x256 | 0.0164 | 0.0020 | 0.0213 | 0.0045 | 1.30x | Orthon faster (noisy) | 0.00e+00 |
| 512x512 | 0.0198 | 0.0015 | 0.0214 | 0.0043 | 1.08x | similar (noisy) | 0.00e+00 |
| 1024x1024 | 0.1235 | 0.0047 | 0.1230 | 0.0037 | 1.00x | ordering unreliable | 0.00e+00 |
| 2048x2048 | 0.8846 | 0.0470 | 0.8660 | 0.0343 | 0.98x | ordering unreliable | 0.00e+00 |

**A prior recorded result did not reproduce.** [`RESULTS.md`](RESULTS.md)
carries a legacy row showing 0.101 ms for Orthon and 0.751 ms for PyTorch at
1024x1024 BF16 — a ~7x gap. Under this harness the two are **0.1235 ms and
0.1230 ms: a dead tie**, at every one of four runs. That legacy row came from
`bench_matmul.py`, which clocks each call individually and so charges per-call
synchronization and Python overhead asymmetrically. The 7x figure is a timing
artefact and is not claimed anywhere. At 256x256 and 512x512 Orthon is nominally
ahead, but both sides are under 25 microseconds there, both rows are flagged
noisy, and the 512x512 ratio moved between 1.01x and 1.31x across runs, landing
on both sides of the 10% threshold — so no BF16 claim is made either.

### Elementwise add

float32, forward. Memory-bandwidth bound on both sides.

| Shape | Orthon median ms | Orthon IQR | PyTorch median ms | PyTorch IQR | Speedup | Verdict | Max rel diff |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 1,000,000 elements | 0.0259 | 0.0025 | 0.0203 | 0.0024 | 0.78x | PyTorch faster | 0.00e+00 |
| 4,000,000 elements | 0.3509 | 0.0145 | 0.3606 | 0.0203 | 1.03x | ordering unreliable | 0.00e+00 |
| 16,000,000 elements | 1.4355 | 0.0694 | 1.4612 | 0.0380 | 1.02x | ordering unreliable | 0.00e+00 |

### Sum reduction

float32, forward. Full reduction to a scalar.

| Shape | Orthon median ms | Orthon IQR | PyTorch median ms | PyTorch IQR | Speedup | Verdict | Max rel diff |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 1,000,000 elements | 0.0293 | 0.0049 | 0.0297 | 0.0056 | 1.01x | ordering unreliable (noisy) | 6.25e-08 |
| 4,000,000 elements | 0.0400 | 0.0052 | 0.0375 | 0.0024 | 0.94x | ordering unreliable | 6.25e-08 |
| 16,000,000 elements | 0.4983 | 0.0246 | 0.5054 | 0.0147 | 1.01x | ordering unreliable | 0.00e+00 |

### FP32 matmul

float32, forward, strict float32 on both sides (TF32 off). Orthon dispatch:
`custom-tiled-float32`.

| Shape | Orthon median ms | Orthon IQR | PyTorch median ms | PyTorch IQR | Speedup | Verdict | Max rel diff |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 256x256 | 0.0419 | 0.0036 | 0.0227 | 0.0022 | 0.54x | PyTorch faster | 5.46e-07 |
| 512x512 | 0.2987 | 0.0097 | 0.0542 | 0.0027 | 0.18x | PyTorch faster | 9.39e-07 |
| 1024x1024 | 2.3121 | 0.0689 | 0.3820 | 0.0552 | 0.17x | PyTorch faster | 1.24e-06 |
| 2048x2048 | 18.6589 | 0.5755 | 2.9375 | 0.3434 | 0.16x | PyTorch faster | 2.20e-06 |

Orthon's handwritten tiled float32 GEMM is 1.8x slower than cuBLAS at
256x256 and 6.4x slower at 2048x2048 - the gap widens with size. This is the
expected result and it is published without qualification.

### Conv2d 3x3

float32, forward, 3x3 stride 1 padding 1, NCHW, cuDNN TF32 off.

| Shape | Orthon median ms | Orthon IQR | PyTorch median ms | PyTorch IQR | Speedup | Verdict | Max rel diff |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| N=128 C=3 32x32 -> 32ch 3x3 | 0.6708 | 0.0345 | 0.2293 | 0.0206 | 0.34x | PyTorch faster | 3.63e-07 |
| N=64 C=32 32x32 -> 32ch 3x3 | 2.6091 | 0.0930 | 0.2207 | 0.0227 | 0.08x | PyTorch faster | 1.05e-06 |
| N=32 C=64 16x16 -> 64ch 3x3 | 1.3143 | 0.0314 | 0.1442 | 0.0067 | 0.11x | PyTorch faster | 8.43e-07 |

Orthon's convolution is 2.9x to 11.8x slower than cuDNN. Numerically it agrees
to under `1.1e-06`; it is simply not a competitive kernel.

## Stable Orthon wins

Rows at >= 10% lower Orthon median with a reliable ordering, canonical run:

| Benchmark | Shape | Speedup | Stability |
| --- | --- | ---: | --- |
| Cross entropy (forward) | 2048x4096 | 3.06x | stable |
| Fused Linear + ReLU | M=128 K=256 N=128 | 2.79x | stable |
| Cross entropy (forward) | 4096x4096 | 1.80x | stable |
| Cross entropy (fwd+bwd) | 4096x4096 | 1.44x | stable |
| Cross entropy (fwd+bwd) | 1024x2048 | 1.44x | stable |
| Cross entropy (fwd+bwd) | 2048x4096 | 1.39x | stable |
| Cross entropy (fwd+bwd) | 512x1024 | 1.38x | stable |
| BF16 matmul | 256x256 | 1.30x | noisy |
| Softmax | 256x512 | 1.24x | stable |

9 of 36. One is flagged noisy. Softmax and BF16 matmul appear here at their
smallest shapes only and reverse or become unreliable at every larger shape,
so neither is claimed anywhere — see the retractions below.

## Ties and unreliable orderings

| Benchmark | Shape | Speedup | Label |
| --- | --- | ---: | --- |
| BF16 matmul | 512x512 | 1.08x | similar |
| Elementwise add | 4,000,000 elements | 1.03x | ordering unreliable |
| Elementwise add | 16,000,000 elements | 1.02x | ordering unreliable |
| Sum reduction | 16,000,000 elements | 1.01x | ordering unreliable |
| Sum reduction | 1,000,000 elements | 1.01x | ordering unreliable |
| BF16 matmul | 1024x1024 | 1.00x | ordering unreliable |
| Softmax | 2048x4096 | 0.99x | ordering unreliable |
| BF16 matmul | 2048x2048 | 0.98x | ordering unreliable |
| Softmax | 512x1024 | 0.98x | ordering unreliable |
| Sum reduction | 4,000,000 elements | 0.94x | ordering unreliable |
| Softmax | 1024x2048 | 0.88x | ordering unreliable |

"Ordering unreliable" means the interquartile speedup band contains 1.00x: the
data does not establish which framework is faster, whatever the medians say.
The sum-reduction rows are the clearest example — `4,000,000 elements` read
0.94x, 1.07x, 0.95x and 1.12x across the four runs, landing on both sides of
parity with overlapping bands every time.

## PyTorch wins

| Benchmark | Shape | Speedup | Orthon is slower by |
| --- | --- | ---: | ---: |
| Conv2d 3x3 | N=64 C=32 32x32 -> 32ch 3x3 | 0.08x | 11.8x |
| Conv2d 3x3 | N=32 C=64 16x16 -> 64ch 3x3 | 0.11x | 9.1x |
| Fused Linear + ReLU | M=1024 K=1024 N=1024 | 0.14x | 7.2x |
| FP32 matmul | 2048x2048 | 0.16x | 6.4x |
| FP32 matmul | 1024x1024 | 0.17x | 6.1x |
| Fused Linear + ReLU | M=512 K=1024 N=512 | 0.17x | 5.7x |
| Fused Linear + ReLU | M=512 K=512 N=512 | 0.18x | 5.6x |
| FP32 matmul | 512x512 | 0.18x | 5.5x |
| Cross entropy (forward) | 8192x512 | 0.25x | 4.1x |
| Conv2d 3x3 | N=128 C=3 32x32 -> 32ch 3x3 | 0.34x | 2.9x |
| Cross entropy (forward) | 1024x2048 | 0.44x | 2.3x |
| Cross entropy (forward) | 512x1024 | 0.46x | 2.2x |
| Fused Linear + ReLU | M=256 K=512 N=256 | 0.53x | 1.9x |
| FP32 matmul | 256x256 | 0.54x | 1.8x |
| Elementwise add | 1,000,000 elements | 0.78x | 1.3x |
| Fused Linear + ReLU | M=256 K=256 N=256 | 0.85x | 1.2x |

16 of 36 — nearly twice the Orthon column. On the general-purpose dense
workloads that dominate real training (FP32 GEMM, convolution), PyTorch wins by
a wide margin, and Orthon does not claim otherwise anywhere. Four of the six
Fused Linear + ReLU shapes are in this table: once the baseline is the
idiomatic `F.linear` spelling, Orthon's fusion advantage survives only at the
smallest shape in the matrix.

## Reproducibility of the headline rows

Four independent full-matrix runs (A, B, C, D — C is the canonical run above),
same session, same build, same machine, with the GPU on mains power and its SM
clock holding 2505 MHz throughout.

| Benchmark | Shape | A | B | C (canonical) | D | Noisy in |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Cross entropy (forward) | 2048x4096 | 2.45x | 2.75x | **3.06x** | 2.47x | 0/4 |
| Cross entropy (forward) | 4096x4096 | 1.63x | 1.73x | **1.80x** | 1.77x | 0/4 |
| Cross entropy (fwd+bwd) | 4096x4096 | 1.36x | 1.41x | **1.44x** | 1.41x | 0/4 |
| Cross entropy (fwd+bwd) | 2048x4096 | 1.42x | 1.35x | **1.39x** | 1.39x | 0/4 |
| Fused Linear + ReLU | M=128 K=256 N=128 | 2.86x | 3.24x | **2.79x** | 3.04x | 3/4 |
| Fused Linear + ReLU | M=256 K=256 N=256 | 0.77x | 0.91x | **0.85x** | 0.90x | 2/4 |
| BF16 matmul | 512x512 | 1.01x | 1.18x | **1.08x** | 1.31x | 3/4 |
| Softmax | 256x512 | 1.16x | 1.14x | **1.24x** | 1.14x | 0/4 |
| FP32 matmul | 2048x2048 | 0.16x | 0.15x | **0.16x** | 0.14x | 0/4 |
| Conv2d 3x3 | N=64 C=32 32x32 | 0.09x | 0.09x | **0.08x** | 0.09x | 0/4 |

The four cross-entropy rows and the two large PyTorch wins reproduce tightly
and are never flagged noisy. The two Fused Linear + ReLU rows are noisy in most
runs: at those sizes both frameworks are dominated by per-call dispatch cost.
Note that `M=256 K=256 N=256` is now below 1.00x in all four runs — under the
earlier three-dispatch baseline this same shape read 1.57x-2.47x, which is the
clearest single demonstration of why that baseline had to go. BF16 straddles
the threshold and is not claimed.

**A note on measurement conditions.** These numbers were taken with the laptop
on mains power. On battery the same machine power-caps this GPU to roughly half
its boost clock, and it does so asymmetrically: in a battery-limited session
the `cross_entropy 2048x4096` row was observed anywhere from 0.57x to 3.30x,
because Orthon's kernels degrade far more than PyTorch's cuBLAS/cuDNN paths do.
Anyone reproducing this matrix on a laptop should confirm the SM clock is
holding under load before trusting a ratio.

## Selected portfolio results

**Two** results are promoted to the README, chosen for stability first and raw
ratio last. Both are an Orthon win in all four runs, are never flagged noisy,
and clear their numerical tolerance by two orders of magnitude or more:

| Result | Orthon | PyTorch eager | Speedup | Across 4 runs | Why this one |
| --- | ---: | ---: | ---: | --- | --- |
| Cross entropy forward, 2048x4096 | 0.175 ms | 0.536 ms | 3.06x | 2.45x-3.06x, 0/4 noisy | Specialized fused loss kernel; exact numerical match (0.00e+00) |
| Cross entropy forward + backward, 4096x4096 | 3.143 ms | 4.523 ms | 1.44x | 1.36x-1.44x, 0/4 noisy | Includes the fused backward; 64 MiB working set, past the 32 MiB L2, so not a cache artefact |

**There is no third card, and that is the honest outcome of this revision.**
The previous third result was Fused Linear + ReLU at `M=256 K=256 N=256`,
published at 1.68x. Under the corrected `F.linear` baseline that shape is a
PyTorch win in all four runs (0.77x-0.91x), so the claim is withdrawn rather
than replaced. The remaining candidates were each rejected for a stated reason:

- `Fused Linear + ReLU M=128 K=256 N=128` (2.79x) — noisy in 3 of 4 runs, and
  the smallest shape in the matrix, where dispatch cost dominates.
- `Cross entropy forward 4096x4096` (1.80x) and
  `Cross entropy fwd+bwd 2048x4096` (1.39x) — both stable and both qualifying,
  but they are the same two workloads as the promoted cards at a second shape,
  and a portfolio section is not improved by saying the same thing twice.
- `Softmax 256x512` (1.24x) and `BF16 matmul 256x256` (1.30x) — the softmax and
  BF16 claims are retracted repository-wide and are not being reinstated on the
  strength of a single small shape whose larger siblings sit at or below parity.

## Limitations

- **One GPU, one machine, one driver.** Everything here is an RTX 4070 Laptop
  GPU on Windows 11 with driver 596.08. A different GPU, a different PyTorch
  build or a laptop under different thermal conditions will produce different
  numbers.
- **These are microbenchmarks.** A single op in a loop with warm caches is not
  a training step. The 2048x4096 float32 logits tensor is exactly 32 MiB and
  fits this GPU's 32 MB L2, so both frameworks read it mostly from cache; the
  `4096x4096` row (64 MiB) is included specifically so the cross-entropy result
  can be checked outside L2, and it still favours Orthon at 1.80x.
- **PyTorch eager only.** No `torch.compile` anywhere — Triton has no Windows
  wheel on this machine. A compiled PyTorch would fuse more aggressively still,
  and would very likely erase what is left of the Linear + ReLU advantage at
  the one shape that keeps it.
- **Laptop clocks are not fixed.** No clock locking is applied. Alternating
  rounds and 11 samples reduce drift; they do not eliminate it. This is why
  sub-30-microsecond rows are labelled noisy rather than trusted. Mains power
  is required for a meaningful run: see the note under
  [reproducibility](#reproducibility-of-the-headline-rows).
- **Forward-only unless the row says otherwise.** Only cross entropy has a
  forward + backward row. Nothing here measures a full training step, an
  optimizer, or a multi-layer model.
- **Coverage is narrow.** Nine benchmarks, 36 cases. Orthon has many more ops
  than this, most of them never compared against PyTorch for speed.

## Reproduction commands

Run from the repository root, in an environment with a CUDA-enabled Orthon
build and a CUDA PyTorch build.

### Reproducing the RTX 4070 results

The published RTX 4070 measurements were taken against an Orthon CUDA build
compiled natively for **SM 89**, so its kernels run as Ada SASS. Orthon does
not require SM 89 — the repository's portable configuration builds PTX that
the driver JITs onto newer architectures, and it runs correctly there — but a
JIT-fed build is not the environment these timings came from, and its numbers
may differ. To reproduce the exact setup, build and install for SM 89 first.

This is the command that produced the benchmarked installation, run from a
Visual Studio 2022 `vcvars64` shell on Windows (`cd` to the repository root
first):

```bat
set SKBUILD_CMAKE_DEFINE=WITH_CUDA=ON;WITH_NCCL=OFF;CMAKE_CUDA_ARCHITECTURES=89
python -m pip install . --force-reinstall --no-deps
```

The same settings on a POSIX shell:

```bash
SKBUILD_CMAKE_DEFINE="WITH_CUDA=ON;WITH_NCCL=OFF;CMAKE_CUDA_ARCHITECTURES=89" \
    python -m pip install . --force-reinstall --no-deps
```

The harness prints the installed extension's real SM targets in its
environment block (`Orthon extension SM`), so a reader can confirm which build
they are measuring before reading any timing.

Then, for the full comparison:

```bash
python benchmarks/compare_pytorch.py
```

### Individual benchmarks

List the benchmarks and every shape each one accepts:

```bash
python benchmarks/compare_pytorch.py --list
```

One benchmark, all of its shapes:

```bash
python benchmarks/compare_pytorch.py --benchmark cross_entropy
```

One benchmark, one shape:

```bash
python benchmarks/compare_pytorch.py --benchmark cross_entropy --shape 2048 4096
python benchmarks/compare_pytorch.py --benchmark fused_linear_relu --shape 256 256 256
python benchmarks/compare_pytorch.py --benchmark bf16_matmul --shape 1024
```

Numerical equivalence checks only, no timing:

```bash
python benchmarks/compare_pytorch.py --check-only
```

The screenshot-ready single-result card:

```bash
python benchmarks/compare_pytorch.py --highlight cross_entropy --shape 2048 4096
```

```text
============================================================
  Orthon vs. PyTorch eager
  Cross entropy (forward)  |  2048x4096  |  float32
  GPU: NVIDIA GeForce RTX 4070 Laptop GPU
============================================================
  Orthon         0.170 ms
  PyTorch        0.532 ms
  Speedup         3.13x
============================================================
  median of 11 alternating samples, 468 iterations each
  max relative difference vs PyTorch: 0.00e+00 (tolerance 1e-05)
============================================================
```

Tighter statistics, machine-readable output, and the separately labelled TF32
and compiled variants:

```bash
python benchmarks/compare_pytorch.py --samples 21 --target-ms 80
python benchmarks/compare_pytorch.py --markdown matrix.md --json matrix.json
python benchmarks/compare_pytorch.py --benchmark fp32_matmul --tf32
python benchmarks/compare_pytorch.py --benchmark fused_linear_relu --torch-compile
```

`python benchmarks/compare_pytorch.py --help` documents every flag. The harness
exits non-zero if any numerical check fails.
