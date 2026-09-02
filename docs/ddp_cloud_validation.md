# Distributed Training: Supported vs Validated

The contract for `orthon.distributed`. The stack is intentionally narrow and
correctness-first: a single-node data-parallel path with local-multiprocessing
and NCCL backends — not production-grade PyTorch DDP parity, and not claimed
to be.

This document keeps two questions strictly separate: **what the implementation
supports** (code that exists, with unit coverage where a single machine can
provide it) and **what has actually been runtime-validated, where**. The CUDA
device-context machinery underneath is described in
[cuda_runtime.md](cuda_runtime.md).

## What the implementation supports

- A DDP wrapper that broadcasts parameters from rank 0 at construction.
- Automatic gradient synchronization through post-accumulate grad hooks, or
  manual `ddp_model.all_reduce_gradients()`.
- Optional bucketed gradient synchronization (`bucket_size_mb`).
- Async work handles for supported collectives:
  - `ddist.all_reduce(tensor, op="sum", async_op=True) -> Work`
  - `ddist.broadcast(tensor, src=0, async_op=True) -> Work`
  - `Work.wait()`, `Work.is_completed()`
- A limited overlap mode for bucketed DDP
  (`DDP(model, bucket_size_mb=..., overlap_grad_sync=True)`): packed bucket
  all-reduces launch asynchronously as buckets become ready, in deterministic
  order, and `backward()` waits before returning so the optimizer always sees
  reduced gradients.
- AMP-DDP integration: ranks synchronize non-finite-gradient decisions and
  average FP32 master gradients, so loss-scale choices stay consistent across
  ranks.

### Backend contract

`backend="local"` — correctness-first local multiprocessing:

- Process-group lifecycle, `broadcast`, `all_reduce(op="sum")`, `barrier`.
- Async collectives return already-completed work handles.
- Serializes tensor payloads through CPU/multiprocessing state.
- The backend behind the CPU DDP parity tests and
  `examples/ddp_local_demo.py`.

`backend="nccl"` — single-node NCCL:

- GPU-resident **contiguous** CUDA `float32`/`float16`/`bfloat16` tensors for
  `broadcast` and `all_reduce(op="sum")`; `barrier` via a tiny dummy
  all-reduce.
- Async collectives launch on a private CUDA stream and synchronize at
  `Work.wait()`.
- DDP gradient synchronization is intentionally `float32`/`float64` only;
  bucketed NCCL DDP packs contiguous `float32` gradients. Low-precision
  gradient averaging stays disabled until dtype-preserving in-place scaling
  exists.

Unsupported NCCL inputs are rejected, and the rejections are tested: CPU
tensors, non-contiguous tensors, unsupported dtypes, tensors on the wrong CUDA
device for the rank, and `all_reduce` ops other than `"sum"`.

### Launcher

```bash
python -m orthon.distributed.run --nproc_per_node N --backend nccl script.py
```

The launcher starts local subprocesses and sets `RANK`, `WORLD_SIZE`,
`LOCAL_RANK`, `LOCAL_WORLD_SIZE` and `ORTHON_DISTRIBUTED_BACKEND`;
`init_process_group()` infers its configuration from those. For NCCL it also
creates a temporary local-file rendezvous for the NCCL unique ID. This is
single-node ergonomics only — there is no multi-node rendezvous system.

## What has been validated, and where

Four distinct validation environments, with different weight:

**1. The final release validation machine (Linux, 2x NVIDIA A100-SXM4-80GB,
CUDA 12.8, NCCL 2.27.3) — a one-time run, and the authoritative multi-GPU
record for this release.**

That run was made on the release-candidate tree as it stood immediately
before one later change: a correction to leaf gradient *ownership* in the
autograd engine, which stopped two distinct leaves that are direct operands
of one same-shape `add` from being handed the same gradient buffer. The
correction touches `src/core/tensor.cpp` alone. It is device-agnostic and
changes no computed gradient value; `DeviceGuard`, `Tensor.to` and the
peer-copy path, the memory pool, the NCCL backend and the DDP transport are
all byte-identical to what ran on the two A100s. The post-correction tree was
revalidated on the development machine — full CPU and CUDA Python suites,
CPU and CUDA native CTest, and the multi-rank local DDP backend including
multi-step overlap — but **it has not itself been rerun on two physical
GPUs.** The two-GPU evidence below therefore stands for the multi-GPU
implementation, not for a rerun of the final tree.

Runtime-validated there, with rank 0 on `cuda:0` and rank 1 on `cuda:1`:

- the full `test_distributed_nccl.py` two-GPU surface: communicator
  init/shutdown, broadcast and all-reduce for float32/float16/bfloat16,
  async collectives with `Work.wait()`, barrier, launcher smoke, and every
  NCCL rejection case (CPU tensors, wrong-device tensors, non-contiguous
  tensors, unsupported dtypes, non-`"sum"` ops);
- DDP training across two GPUs in every synchronization mode — explicit
  `all_reduce_gradients()`, blocking auto-sync, bucketed, and bucketed with
  `overlap_grad_sync=True` — including **multi-step** overlap training with
  plain `optimizer.zero_grad()` between steps, verifying that new bucket
  collectives launch on every backward and that gradients and parameters
  stay synchronized across ranks, with clean shutdown and no deadlocks;
- AMP-DDP finite/non-finite gradient-decision synchronization across two
  ranks.

**2. The development machine (Windows, one RTX 4070 Laptop GPU) — the ongoing
local matrix.** Runtime-validated here:

- the full local-backend and DDP CPU test files
  (`test_distributed_skeleton.py`, `test_distributed_local_backend.py`,
  `test_distributed_ddp.py`, `test_distributed_ddp_training.py`);
- CUDA `world_size=1` DDP smoke (`test_distributed_ddp_cuda.py`);
- NCCL backend registration and its clean rejection when runtime support is
  unavailable (`test_distributed_skeleton.py`). The NCCL binding and backend
  source were compile-verified in the CUDA+NCCL build.

**Not** runtime-validated on the one-GPU development machine: any NCCL
collective execution and any NCCL-specific input-validation case — every NCCL
execution test in `test_distributed_nccl.py`, negative cases included, gates
on a CUDA+NCCL build with at least two visible GPUs — and any
`world_size >= 2` GPU path. Those tests skip on one GPU, explicitly; on the
final release machine (environment 1) they all ran.

**3. A single-node cloud machine (Lambda, multi-GPU) — an older recorded
one-time validation of an EARLIER project state, before the final release
corrections.** The NCCL backend and DDP paths were run there across 2, 4 and
8 GPUs via `tests/python/test_distributed_nccl.py`:

| Scope | Coverage exercised |
| --- | --- |
| 2 GPUs | init/shutdown, broadcast, async broadcast, all-reduce, async all-reduce, barrier, DDP explicit sync, DDP auto-sync, bucketed DDP, overlap DDP, launcher smoke, negative cases |
| 4 GPUs | init/shutdown, broadcast, all-reduce, barrier, explicit DDP gradient average |
| 8 GPUs | init/shutdown, broadcast, all-reduce, barrier, explicit DDP gradient average |

The 2-GPU scope has since been superseded by the release validation run
(environment 1). The 4- and 8-GPU scope has **not**
been rerun on the final corrected tree — the `>=4`/`>=8`-GPU test gates
skipped on the two-GPU release machine — so multi-GPU coverage beyond two
devices rests on this earlier record only. Those tests remain in the tree and
skip unless CUDA, NCCL and the required GPU count are visible; anyone with
the hardware can re-run them.

**4. CI — CPU only.** The local backend and DDP logic run in CI; nothing
CUDA- or NCCL-related does ([testing.md](testing.md#continuous-integration)).
The CPU tier includes a multi-step local-backend overlap regression
(`test_distributed_ddp_training.py`) that catches a missing per-backward
gradient synchronization without any GPU.

The distinction matters: statements about two-GPU NCCL and DDP behaviour rest
on the release validation run of environment 1, whose relationship to the
final tree is stated precisely above; statements about 4/8 GPUs rest on the
earlier cloud record and the code's tested contracts. Nothing in this
repository simulates a GPU count it does not have.

## Usage

Blocking automatic DDP:

```python
ddp_model = DDP(model, sync_grads=True)
loss = compute_loss(ddp_model)
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

Bucketed, with limited overlap:

```python
ddp_model = DDP(model, sync_grads=True, bucket_size_mb=25, overlap_grad_sync=True)
loss = compute_loss(ddp_model)
loss.backward()   # waits before returning; optimizer sees reduced gradients
optimizer.step()
optimizer.zero_grad()
```

Overlap synchronization state is **per-backward**: each new backward pass
starts its bucket bookkeeping fresh, so the ordinary
`backward() / step() / zero_grad()` loop above is the supported multi-step
pattern. Calling the optimizer's own `zero_grad()` is sufficient —
`ddp_model.zero_grad()` also works, but is not required to reset overlap
state between steps. Multi-step overlap training is regression-tested on the
CPU local backend and was validated on real 2-GPU NCCL hardware.

Manual synchronization and raw async collectives:

```python
ddp_model = DDP(model, sync_grads=False)
loss.backward()
ddp_model.all_reduce_gradients()

work = ddist.all_reduce(tensor, op="sum", async_op=True)
# independent work that does not read `tensor`
work.wait()
```

### Benchmark driver

```bash
python -m orthon.distributed.run --nproc_per_node 2 --backend nccl \
  benchmarks/benchmark_distributed_ddp.py --mode blocking
python -m orthon.distributed.run --nproc_per_node 2 --backend nccl \
  benchmarks/benchmark_distributed_ddp.py --mode bucketed --bucket-size-mb 25
python -m orthon.distributed.run --nproc_per_node 2 --backend nccl \
  benchmarks/benchmark_distributed_ddp.py --mode overlap --bucket-size-mb 25
```

`--warmup-iters`, `--timed-iters`, `--batch-size`, `--hidden-size` and
`--depth` scale the synthetic MLP. The benchmark prints one parseable rank-0
line with `world_size`, `mode`, `bucket_size_mb`, `avg_step_ms` and
`avg_backward_ms`.

### Build

```bash
python -m pip install -e . --no-build-isolation \
  -Ccmake.define.WITH_CUDA=ON \
  -Ccmake.define.WITH_NCCL=ON
```

or `cmake --preset cuda-nccl-release`. NCCL builds require NCCL headers and
library discoverable by CMake; `orthon.nccl_enabled()` reports the result.

## Limitations and non-claims

- No multi-node distributed training, and no multi-node rendezvous.
- No production DDP parity, fault tolerance, or elastic launch behaviour.
- No sharded or distributed checkpointing
  ([checkpoint_format.md](checkpoint_format.md) is single-file, local).
- No full communication/compute overlap across arbitrary model graphs — only
  the deterministic bucketed overlap mode described above.
- No NCCL collectives beyond `broadcast`, `all_reduce(op="sum")` and the
  barrier path.
- AMP-DDP synchronizes finite-gradient decisions and FP32 master gradients;
  it is not a full production distributed-AMP implementation.
- DDP low-precision gradient averaging is intentionally disabled.
