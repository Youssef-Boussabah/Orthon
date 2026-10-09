# Testing

Orthon has five test layers. Specifications define behavioral contracts;
regressions preserve historical defects, parity and composed workloads;
architecture fences protect source dependencies, ownership and package boundaries.
Overlapping behavioral coverage across layers is intentional.

| Layer | Owner | Selection | Responsibility |
| --- | --- | --- | --- |
| native | `tests/cpp` | CTest, optionally `-L <domain>` | Native ownership, resources, kernels, autograd and API contracts |
| regression | Most `tests/python/test_*.py` | `python -m pytest -q -m regression` | Public regressions, reference parity, workloads and hardware edge cases |
| specification | `tests/spec` | `python -m pytest -q -m spec` | Supported/unsupported behavioral contracts |
| architecture | Modules in `tests/support/layers.py` | `python -m pytest -q -m architecture` | Source/AST, ownership, security, build and package fences |
| package smoke | `tests/smoke/package_smoke.py` | Standalone script against installed artifact | Runtime-only wheel/sdist installation contract |

Native and package smoke are outside normal pytest collection. The default
`python -m pytest -q` still runs the entire pytest suite. Each collected case has
exactly one primary `regression`, `spec` or `architecture` marker. Strict markers
and importlib mode are enabled in `pyproject.toml`. The three primary selections
are disjoint and their union equals full collection. Collected paths, function
names, parametrization and native `Suite.Case` names remain stable.

## Prepare the package explicitly

Use an installed wheel, a deliberate editable install, or an explicit complete
Python package/extension stage. Tests never guess a `build/` directory when import
fails. The caller chooses the package intentionally; keep profiles in separate
build directories. The plain CMake build provides `build/orthon` for an explicitly
chosen `PYTHONPATH=build` run.

For a plain CPU build with Python bindings and no native test dependency:

```bash
cmake -S . -B /tmp/orthon-cpu -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=OFF -DWITH_NCCL=OFF -DBUILD_TESTING=OFF -DBUILD_PYTHON_BINDINGS=ON -DPYTHON_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')"
cmake --build /tmp/orthon-cpu --config Release
export PYTHONPATH=/tmp/orthon-cpu
python -c 'import orthon; print(orthon.__file__); print(orthon._C.__file__); print(orthon.cuda_enabled())'
```

Both paths should be under `/tmp/orthon-cpu/orthon`. The build stages runtime
Python sources alongside `_C`; ordinary rebuilds update edits and add modules.
After removing a Python source, use a fresh build directory to avoid stale copies.
For CUDA, choose a separate directory and `-DWITH_CUDA=ON`, with a compatible
toolkit and target architecture. `BUILD_PYTHON_BINDINGS=OFF` builds no Python stage.
Wheel installation continues to use scikit-build-core's separate package path.

```bash
python -m pip install "torch>=2.0" --index-url https://download.pytorch.org/whl/cpu
python -m pip install '.[test]' -C cmake.define.WITH_CUDA=OFF -C cmake.define.WITH_NCCL=OFF
ORTHON_TEST_PROFILE=cpu python -m pytest -q
```

`ORTHON_TEST_PROFILE` is optional. Unset accepts ordinary developer runs. When
supplied, it verifies the loaded extension before collection:

| Expected profile | `cuda_enabled()` | `nccl_enabled()` |
| --- | --- | --- |
| `cpu` | False | False |
| `cuda` | True | False |
| `cuda-nccl` | True | True |

Invalid spellings and mismatches are actionable pytest usage errors. The normal
pytest header reports package/extension paths, flags, expected profile and Python
version; omit `-q` to see it. Compile profile does not assert usable hardware.

```bash
ORTHON_TEST_PROFILE=cpu python -m pytest -q -m architecture
ORTHON_TEST_PROFILE=cpu python -m pytest -q -m spec
ORTHON_TEST_PROFILE=cpu python -m pytest -q -m regression
ORTHON_TEST_PROFILE=cpu python -m pytest --collect-only -q
```

Use `ORTHON_TEST_PROFILE=cuda` with an intentionally prepared CUDA package for
physical validation. `PYTHONPATH=/absolute/stage` is an explicit caller choice.
Subprocess specifications retain deliberate child environments and bounded
process-tree supervision, and inherit the selected package.

## Native registration and selection

One source remains one executable; `gtest_discover_tests` registers every compiled
case individually with a 300-second timeout. Explicit CMake domain lists are
disjoint and fail-closed for missing registered sources. Architecture tests reject
unregistered `.cpp`/`.cu` sources. No globbing or executable merger is used.
Each discovered case carries `native` plus exactly one domain:

| Domain | Executables |
| --- | --- |
| core | `test_tensor`, `test_tensor_core`, `test_float_formats`, `test_identity` |
| runtime | `test_allocator`, `test_runtime_context`, `test_async_runtime`, `test_device_guard`, `test_dispatcher`, CUDA-only `test_async_cuda` |
| operators | `test_backends`, `test_ops_binary`, `test_ops_shape`, `test_ops_reduce`, `test_reduction_subsystem`, `test_ops_unary`, `test_ops_to`, `test_matmul_planner`, `test_ops_matmul`, `test_indexing`, `test_ops_scalar` |
| autograd | `test_autograd_engine` |
| distributed | `test_distributed_runtime`, `test_distributed_cuda` |
| compatibility | `test_compatibility_surface` |

Conditional compilation determines native discovery. Use CTest listing to inspect
the selected build; record exact discovery/status totals with each validation run.

```bash
cmake --preset cpp-cpu-release
cmake --build --preset cpp-cpu-release
ctest --preset cpp-cpu-release -N
ctest --preset cpp-cpu-release
ctest --preset cpp-cpu-release -L runtime
ctest --preset cpp-cpu-release -L compatibility
```

The CUDA equivalent uses `cpp-cuda-release`. These native presets turn bindings
off; validation can separately enable them in a fresh build. GoogleTest is required
only with `BUILD_TESTING=ON`: CMake prefers an installed compatible copy, otherwise
fetches the pinned revision when allowed, and fails if unavailable. Test presets
use `noTestsAction: error`. Package builds force testing off and need no GoogleTest.

## Shared support and source tests

`tests/conftest.py` owns primary markers, expected-profile verification and
provenance. Domain fixtures stay in specification conftests. `tests/support`
contains the layer manifest, source readers, Orthon device probe,
scoped example loader and current compatibility consumer boundaries. Collected
modules do not serve as helper libraries. Domain workload services remain in `tests/spec`,
including AMP state helpers, token models and distributed parameter cases.

The shared regression CUDA probe performs an actual Orthon allocation/transfer
and returns False for an unavailable runtime. PyTorch CUDA is a separate gate
when used as an oracle. Specification `cuda_device` remains stronger: known
unavailability skips, unexpected errors fail. NCCL and multi-GPU tests keep real
build/device gates. Source fences complement behavior; source text does not prove
correct computation. Private diagnostics through `orthon._diagnostics` remain available for source
fences and benchmark observations; they are not public application APIs. No xfail debt hides
unsupported CUDA cases.

## CPU CI and artifact smoke

`.github/workflows/ci.yml` retains native Linux/Windows jobs and Python 3.10/3.13
CPU jobs. Each Python job installs one CPU package, then executes architecture,
specification and regression steps with `ORTHON_TEST_PROFILE=cpu`. No pass counts
are hard-coded and no layer ignores failures. There is no CUDA CI job on hosted
hardware.

The separate package job builds a CPU wheel and sdist, inspects artifacts,
installs each in a clean environment, checks torch is absent, and runs the
standalone smoke without pytest. Only `python/orthon` is the wheel package;
repository tests/support and GoogleTest artifacts must be absent. Tests are
excluded from the runtime sdist too. Metadata architecture cases can skip on a
bare stage but execute against an installed package.

```bash
python -m build -C cmake.define.WITH_CUDA=OFF -C cmake.define.WITH_NCCL=OFF
python -m venv /tmp/orthon-wheel-smoke
/tmp/orthon-wheel-smoke/bin/python -m pip install dist/*.whl
/tmp/orthon-wheel-smoke/bin/python tests/smoke/package_smoke.py
```

## Documentation fences

`tests/python/test_documentation_architecture.py` checks documentation architecture.
`tests/support/documentation.py` defines the explicit current-doc and major public
export manifests. The fences check file links, explicit repository paths, command
modules/scripts, known retired tokens, exposed Python returns and example syntax.
The manifest matches the complete current guide set; wording and volatile pass
counts are not contracts.

## Validation requirements

CPU CI runs native, architecture, specification, regression and package checks.
There is no hosted CUDA job. Select an explicit installed package for each
physical hardware profile; second-GPU tests require two usable devices. Known
unsupported CUDA cast pairs and CPU-only fill paths remain visible specification
failures without xfail. Record counts, messages and the failure ledger with the run.

NCCL tests require the dependencies and hardware described in
[distributed](distributed.md). Native sanitizer campaigns should report the
instrumentation, exclusions and tested revision. Checkpoint byte stability is a
separate contract described in [checkpoint format](checkpoint_format.md).
