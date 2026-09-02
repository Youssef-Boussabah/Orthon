# Testing

How Orthon is tested, what each layer actually proves, and where the honest
limits are. Two validation environments produced the numbers below and are
kept distinct throughout:

- **the development machine** (Windows 11, MSVC 19.44, CUDA 13.2, one
  RTX 4070 Laptop GPU, plus WSL Ubuntu 24.04 for Linux) — the ongoing local
  matrix, re-run on the current tree;
- **the release validation machine** (Linux, 2x NVIDIA A100-SXM4-80GB,
  driver 580.126.16, CUDA 12.8, GCC 13.3, Python 3.12, NCCL 2.27.3) — a
  one-time run covering everything the single-GPU machine must skip. It ran
  the release-candidate tree as it stood immediately before the final
  leaf-gradient-ownership correction; see
  [the multi-GPU boundary](#cuda-and-multi-gpu-boundary).

The system under test is described in [architecture.md](architecture.md); the
per-operation coverage the suites pin is tabulated in
[cuda_coverage.md](cuda_coverage.md).

## Layers

| Layer | Where it runs | What it proves |
| --- | --- | --- |
| Native C++ (GoogleTest + CTest) | Windows/MSVC, Linux/GCC, locally and in CI | The C++ tensor engine, autograd node contracts, device layer and the `orthon::` API spelling, with no Python in the picture |
| Python CPU (pytest) | Locally and in CI (3.10, 3.13) | The full public API against a CPU PyTorch reference build |
| Python CUDA (pytest) | **Locally only** | Every CUDA kernel, device-context and mixed-precision path |
| Packaging smoke | Locally and in CI | A wheel/sdist install needs only the declared runtime dependencies |

## Native C++ suite

Twelve executables built from `tests/cpp/`, each `TEST()` registered as its own
CTest case via `gtest_discover_tests`, so a failure names the test rather than
the binary.

```bash
cmake --preset cpp-cpu-release
cmake --build --preset cpp-cpu-release
ctest --preset cpp-cpu-release
```

and the CUDA equivalent:

```bash
cmake --preset cpp-cuda-release
cmake --build --preset cpp-cuda-release
ctest --preset cpp-cuda-release
```

Both presets pin `BUILD_PYTHON_BINDINGS=OFF`, so the native suite does not
depend on the Python extension being built.

### Counts

`tests/cpp/` declares 194 `TEST()` cases. Not all of them exist in every
configuration: three are compiled only in a CPU-only build (they assert that a
CUDA target is *rejected*) and twelve only in a CUDA build, so the discovered
totals differ by design.

| Configuration | Discovered | Passed | Skipped | Failed |
| --- | --- | --- | --- | --- |
| Windows MSVC, CPU | 182 | 182 | 0 | 0 |
| Windows MSVC, CUDA (1 GPU) | 191 | 188 | 3 | 0 |
| Linux GCC 13.3 (WSL Ubuntu 24.04), CPU | 182 | 182 | 0 | 0 |

The row below is **historical**: it was recorded on the release-candidate
tree, which declared 189 `TEST()` cases — five fewer than the current tree,
whose extra five are the CPU-only leaf-gradient-ownership regressions in
`test_ops_binary.cpp`.

| Configuration | Discovered | Passed | Skipped | Failed |
| --- | --- | --- | --- | --- |
| Linux GCC 13.3, CUDA (2x A100, release-candidate tree) | 186 | 186 | 0 | 0 |

The three single-GPU CUDA skips are the `DeviceGuard` cases that need **two or
more** GPUs — `SelectsASecondDeviceAndRestoresTheFirst`,
`NestedGuardsUnwindInOrder` and `ExceptionUnwindingRestoresAcrossDevices`.
They call `GTEST_SKIP()` after querying `cudaGetDeviceCount`, so they are
reported as skipped, never as passed. On the 2-GPU release validation run all
three executed against a real second device and passed, which is the 186/186
historical row. There is no other skip in any configuration, and no disabled
test.

`tests/cpp/test_identity.cpp` is a contract test rather than a behavioural one:
it names every public type and function through `orthon::`, and defines global
types and variables of the same names, so it fails to **compile** if anything
in a public header leaks back into the global namespace. It doubles as a
header-consumer smoke test for the `<orthon/...>` include spelling.

## GoogleTest dependency

`BUILD_TESTING` (from `include(CTest)`) is the switch.

- **ON** — a compatible installed GoogleTest is preferred
  (`find_package(GTest CONFIG)`); otherwise the pinned release in
  `ORTHON_GTEST_COMMIT` is fetched with `FetchContent`. That pin is an
  immutable commit, not a tag, so an upstream retag cannot change what a build
  downloads; `ORTHON_GTEST_VERSION` records the matching release for humans.
- **ON with `-DORTHON_FETCH_GTEST=OFF` and no installed GoogleTest** —
  configuration **fails** with an actionable message. It never succeeds while
  registering zero tests.
- **OFF** — no GoogleTest is required, looked for, or downloaded.
- **scikit-build-core (`pip install .`)** — `BUILD_TESTING` is forced OFF, so
  installing the Python package never pulls in a C++ test framework.

`ctest` exits 0 when it finds no tests at all, so the test presets set
`noTestsAction: error`; an empty suite is a failure, not a silent pass.

## Python suites

```bash
python -m pytest -q            # testpaths comes from pyproject.toml
```

| Configuration | Passed | Skipped | Failed |
| --- | --- | --- | --- |
| CPU build | 2186 | 889 | 0 |
| CUDA build (RTX 4070 Laptop, 1 GPU) | 3397 | 60 | 0 |

The row below is **historical**, recorded on the release-candidate tree before
the leaf-gradient-ownership regressions were added:

| Configuration | Passed | Skipped | Failed |
| --- | --- | --- | --- |
| CUDA+NCCL build (2x A100, release-candidate tree) | 3426 | 20 | 0 |

The CPU skip count is large because every CUDA-gated test skips there; that is
the intended behaviour, not missing coverage. Of the 60 skips on one GPU, 40
are `>=2`-device gates: on the 2-GPU machine those run and the skip count
drops to 20, which is the historical row above. The remaining 20 are
hardware/configuration gates (fourteen need four or eight GPUs, the rest are
CPU-build-only cases, a dataset-availability gate, and one documented
pre-existing skip).

Two files in `tests/python/` are audits rather than behavioural tests:

- `test_build_architecture.py` — the CMake source lists match the tree on disk
  exactly (no globs, no dead entries, no missing ones); every public header
  declares inside `namespace orthon` and opens no other namespace; every
  project macro and environment variable carries the `ORTHON_` prefix; the
  CMake and Python versions agree; no file is an empty placeholder; and the
  branding asset is plain geometry with no provenance metadata.
- `test_package_identity.py` — `import orthon` resolves the whole public
  surface and every submodule sits under the one package, the extension is
  `orthon._C`, the checkpoint identifier stays independent of package
  branding, and the built distribution's metadata is accurate — including the
  README as its long description, and no author, URL or license claimed that
  does not exist.

## Continuous integration

`.github/workflows/ci.yml` runs three CPU jobs on `push`, `pull_request` and
`workflow_dispatch`:

| Job | Runner | Covers |
| --- | --- | --- |
| `cpp` | `ubuntu-latest`, `windows-latest` | Configure, build and `ctest` the native suite through the same presets used locally |
| `python` | `ubuntu-latest`, Python 3.10 and 3.13 | Full Python suite against CPU-only PyTorch |
| `package` | `ubuntu-latest` | sdist + wheel build, clean-environment install, torch-free runtime smoke |

Python 3.10 is the declared minimum (`requires-python` in `pyproject.toml`);
3.13 is the current validated upper end.

The Python job installs PyTorch from `https://download.pytorch.org/whl/cpu`.
torch is a **reference** library for the parity tests, not a runtime
dependency, and a CPU wheel is what a GPU-less runner can honestly use.

## CUDA and multi-GPU boundary

**There is no CUDA job in CI, deliberately.** GitHub-hosted runners have no
NVIDIA GPU. A "CUDA" job on one could only compile CPU sources, or build CUDA
code it can never run — either would report green while proving nothing. No
permanently-queued self-hosted job has been added either, because no
self-hosted CUDA runner exists.

CUDA validation is therefore **manual**, in two tiers.

The ongoing local tier is the single-GPU development machine (RTX 4070
Laptop), re-run on the current tree:

- the full CUDA Python suite (3397 passed / 60 skipped),
- the CUDA native build,
- the CUDA native CTest run (191 discovered / 188 passed / 3 skipped),
- single-GPU device-context and transfer behaviour.

The release validation tier is a one-time run on a Linux machine with two
A100-SXM4-80GB GPUs (CUDA 12.8, NCCL 2.27.3). It ran the release-candidate
tree as it stood immediately before the final leaf-gradient-ownership
correction — a `src/core/tensor.cpp`-only, device-agnostic change that alters
no computed gradient value and leaves `DeviceGuard`, the peer-copy path, the
memory pool, the NCCL backend and the DDP transport byte-identical. **The
post-correction tree has not itself been rerun on two physical GPUs**; it was
revalidated on the development machine and through the multi-rank local DDP
backend. Full reasoning:
[ddp_cloud_validation.md](ddp_cloud_validation.md). Recorded there:

- the full CUDA Python suite — 3426 passed / 20 skipped / 0 failed;
- the native CUDA CTest run — 186 discovered / 186 passed / 0 skipped;
- every previously single-GPU-skipped `>=2`-device case executed for real:
  the three `DeviceGuard` tests, exact `cuda:0`/`cuda:1` placement, cross-GPU
  `Tensor.to` in both directions, cross-GPU autograd, `map_location` to a
  second GPU, per-device memory-pool isolation, the NCCL collective and
  rejection tests, and multi-step DDP in blocking, bucketed and overlap
  modes.

Both tiers are **correctness** tiers, and the rule that matters is narrow and
absolute: **no published timing is attributed to the 2x A100-SXM4-80GB
correctness machine.** Every current performance number — the whole
Orthon-vs-PyTorch comparison and everything quoted in the README — was
measured on the RTX 4070 Laptop machine
([profiles/PYTORCH_COMPARISON.md](../profiles/PYTORCH_COMPARISON.md)).
[profiles/RESULTS.md](../profiles/RESULTS.md) additionally keeps a small set
of clearly labelled **historical** rows from an older single A100-SXM4-40GB
cloud machine, measured against an earlier project state; those are marked as
history and are not part of any current claim. The 40GB historical benchmark
machine and the 2x 80GB correctness machine are different systems and are
never merged.

**What stays outside the final release validation:** the `>=4`- and
`>=8`-GPU test gates (that machine has two GPUs; an earlier project state has
a recorded 2/4/8-GPU cloud run, which is historical evidence for an earlier
tree and is never presented as validation of the 0.1.0 release tree), and any
permanently repeatable GPU job — the A100 run was one-time, not a runner. On single-GPU machines every `>=2`-device
case still skips explicitly; nothing fakes a second device. The distributed
layer's supported-versus-validated contract is in
[ddp_cloud_validation.md](ddp_cloud_validation.md).

One consequence of that validation now lives in the CPU tier: the DDP
overlap-mode synchronization state is per-backward, and
`tests/python/test_distributed_ddp_training.py` contains a multi-step
local-backend overlap regression (world size 2, `optimizer.zero_grad()`
between steps, rank-distinct shards) that fails loudly if any backward stops
launching bucket collectives — GPU-class distributed-correctness coverage
that runs in the ordinary CPU `python` job.

## Checkpoint format

The on-disk checkpoint/state format, its safety properties and its validation
are documented separately in
[`docs/checkpoint_format.md`](checkpoint_format.md). Three test files cover it:
`tests/python/test_checkpoint_format.py` (container, raw byte codec,
determinism, corruption, path safety), `tests/python/test_checkpoint.py`
(model/optimizer round-trips, resume equivalence, `map_location`) and
`tests/cpp/test_float_formats.cpp` (the native low-precision conversions the
format's dtype fidelity rests on).
