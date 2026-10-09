# Contributing to Orthon

## Development setup

Use Python 3.10 or later, a C++17 compiler, CMake and Ninja. From the repository
root, install the CPU package and test extras explicitly:

```bash
python -m venv .venv
# Activate the environment using your shell's normal activation command.
python -m pip install '.[test]' -C cmake.define.WITH_CUDA=OFF -C cmake.define.WITH_NCCL=OFF
ORTHON_TEST_PROFILE=cpu python -m pytest -q -m architecture
ORTHON_TEST_PROFILE=cpu python -m pytest -q -m spec
ORTHON_TEST_PROFILE=cpu python -m pytest -q -m regression
cmake --preset cpp-cpu-release
cmake --build --preset cpp-cpu-release
ctest --preset cpp-cpu-release
```

Choose one installed package or an explicit complete package/extension stage for
each profile. See [testing](docs/testing.md) for package selection, Windows
toolchain setup, native domains and optional CUDA/NCCL requirements.

## Tests and review

Specifications own behavioral contracts; regression tests preserve historical
defects, parity and workloads; architecture tests protect dependencies, ownership
and repository boundaries. Native tests and installed-package smoke complete the
five-layer architecture. Behavioral changes need behavioral/spec coverage;
architecture ownership changes need architecture coverage. Classify new pytest
modules in the existing layer manifest, and retain old case identities and honest
failure/skip accounting.

Describe the problem, resulting behavior, relevant checks and remaining limits
when requesting review. Update current documentation when the public API or an
ownership boundary changes. Preserve development history in Git; current guides
should describe current behavior.

## CUDA, NCCL and measurements

Compilation flags do not prove usable hardware. Run physical CUDA tests with an
explicit CUDA package and report the device/profile used. Hosted CI is CPU-only.
Genuine NCCL validation requires NCCL dependencies and suitable hardware;
injected transports establish ownership and ordering independently. Do not
hide unsupported profiles, failures or hardware limitations.

Follow the existing [benchmark methodology](docs/benchmarking.md): establish
correctness before measurement, record environment and synchronization policy,
and keep performance observations separate from correctness gates. Generated
timing/profiler results are local artifacts, not source or performance promises.

## Repository hygiene and formatting

Do not commit generated builds, binaries, caches, datasets, checkpoints, profiler
captures or result files. Store review reports, test evidence and build scratch
outside the project directory. `uv.lock` is local resolver state and is not release
input. Keep text UTF-8 without BOM and use LF endings; ordinary source and
repository files are non-executable. Invoke Python scripts through Python or
module execution.

Follow the existing C++ style and `.clang-format`, and the surrounding Python
style. Keep formatting changes scoped to the work; this project does not require
a new Python formatter or whole-tree reformatting. Preserve intentional assets
and test fixtures.

Read [LICENSE](LICENSE). Preserve applicable copyright and license notices,
attribution and Git history; do not alter them to make a contribution appear original.
