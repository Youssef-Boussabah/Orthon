"""Explicit build inputs, public namespace isolation and repository identity.

Native test_identity.cpp also enforces namespace isolation at compile time."""
from tests.support.source import REPO_ROOT, read_text as _read, strip_cpp_comments as _strip_c_comments, cmake_list as _cmake_list
from pathlib import Path
import os
import re

import pytest

REPO = REPO_ROOT

pytestmark = pytest.mark.skipif(
    not os.path.isfile(os.path.join(REPO, "CMakeLists.txt")),
    reason="source tree not available next to the test file",
)

SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", "build", "dist",
             "_skbuild", "_deps", ".idea", ".vscode"}
BINARY_EXT = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".zip", ".pyd", ".lib",
              ".obj", ".exe", ".dll", ".so", ".a", ".dylib", ".ncu-rep",
              ".nsys-rep", ".qdrep", ".sqlite", ".ttf", ".otf", ".woff",
              ".woff2"}

# Environment variables the runtime reads that are not Orthon's to name: the
# standard distributed-launcher contract and the CUDA toolkit's own.
EXTERNAL_ENV_VARS = {"RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
                     "MASTER_ADDR", "MASTER_PORT", "CUDA_PATH",
                     "CUDA_VISIBLE_DEVICES"}




def _walk(*subdirs):
    """Every tracked file under `subdirs` (or the whole tree when empty)."""
    roots = [os.path.join(REPO, s) for s in subdirs] or [REPO]
    for root in roots:
        for dirpath, dirs, files in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
            for name in sorted(files):
                full = os.path.join(dirpath, name)
                yield os.path.relpath(full, REPO).replace("\\", "/"), full


def _text_files():
    for rel, full in _walk():
        if os.path.splitext(rel)[1].lower() in BINARY_EXT:
            continue
        try:
            with open(full, encoding="utf-8") as fh:
                yield rel, fh.read()
        except UnicodeDecodeError:                      # pragma: no cover
            pytest.fail("%s is not valid UTF-8" % rel)




def _cmake_source():
    """The root CMakeLists.txt with comments stripped."""
    return re.sub(r"(?m)#[^\n]*", "", _read("CMakeLists.txt"))




# Explicit CMake source lists
class TestSourceLists:
    def test_no_source_globs_remain(self):
        """A glob does not re-run configure when a file is added or deleted."""
        body = _cmake_source()
        assert not re.search(r"(?i)\bfile\s*\(\s*GLOB", body), (
            "CMakeLists.txt must list sources explicitly, never glob them")

    def test_core_source_list_is_exactly_the_live_cpp_tree(self):
        listed = sorted(_cmake_list("ORTHON_CORE_SOURCES"))
        on_disk = sorted(rel for rel, _ in _walk("src") if rel.endswith(".cpp") and not rel.startswith("src/backends/cuda/"))
        assert listed == on_disk, (
            "ORTHON_CORE_SOURCES disagrees with src/**.cpp\n"
            "  listed not on disk: %s\n"
            "  on disk not listed: %s"
            % (sorted(set(listed) - set(on_disk)),
               sorted(set(on_disk) - set(listed))))

    def test_cuda_source_list_is_exactly_the_live_cu_tree(self):
        listed = sorted(_cmake_list("ORTHON_CUDA_SOURCES"))
        on_disk = sorted(rel for rel, _ in _walk("src") if rel.endswith(".cu") or rel.startswith("src/backends/cuda/") and rel.endswith(".cpp"))
        assert listed == on_disk, (
            "ORTHON_CUDA_SOURCES disagrees with src/**.cu\n"
            "  listed not on disk: %s\n"
            "  on disk not listed: %s"
            % (sorted(set(listed) - set(on_disk)),
               sorted(set(on_disk) - set(listed))))

    def test_every_listed_source_exists(self):
        for rel in _cmake_list("ORTHON_CORE_SOURCES") + _cmake_list("ORTHON_CUDA_SOURCES"):
            assert os.path.isfile(os.path.join(REPO, rel)), rel

    def test_native_test_list_matches_the_test_sources(self):
        domains = _cmake_list("ORTHON_TEST_DOMAINS")
        assert set(domains) == {"core", "runtime", "operators", "autograd", "distributed", "compatibility"}
        groups = [_cmake_list("ORTHON_TESTS_" + domain.upper()) for domain in domains]
        targets = [target for group in groups for target in group]
        assert len(targets) == len(set(targets)), "native domains overlap"
        listed = sorted(target for target in targets if target != "test_async_cuda")
        assert "test_async_cuda" in _cmake_list("ORTHON_TESTS_RUNTIME")
        assert sorted(targets) == sorted(Path(rel).stem for rel, _ in _walk("tests/cpp") if rel.endswith((".cpp", ".cu")))
        body = _cmake_source()
        assert 'PROPERTIES TIMEOUT 300 LABELS "native;${domain}"' in body
        assert 'if(_t IN_LIST ORTHON_NATIVE_TESTS)' in body
        assert 'if(_ORTHON_CUDA_ENABLED)' in body
        assert 'orthon_add_cpp_test(test_async_cuda tests/cpp/test_async_cuda.cu runtime)' in body
        on_disk = sorted(
            os.path.splitext(os.path.basename(rel))[0]
            for rel, _ in _walk("tests/cpp") if rel.endswith(".cpp"))
        assert listed == on_disk


# No meaningless empty files
def test_no_zero_byte_source_or_header_files():
    empty = [rel for rel, full in _walk()
             if os.path.getsize(full) == 0
             and os.path.splitext(rel)[1] in
             (".py", ".h", ".hpp", ".cuh", ".cpp", ".cu", ".md", ".toml",
              ".json", ".yml", ".yaml", ".cmake", ".txt", ".svg")]
    assert empty == [], "zero-byte placeholder files: %s" % empty


# C++ namespace contract
class TestNamespaceContract:
    """Every public header declares inside `namespace orthon`.

    The compile-time half of this lives in tests/cpp/test_identity.cpp, which
    defines global types and variables of the same names and therefore fails to
    build if any of them leak back out.
    """

    HEADERS = None

    def _headers(self):
        if TestNamespaceContract.HEADERS is None:
            TestNamespaceContract.HEADERS = [
                (rel, _read(rel)) for rel, _ in _walk("include")
                if rel.endswith((".h", ".hpp", ".cuh")) and not rel.startswith(("include/orthon/ops/cpu/", "include/orthon/ops/cuda/"))
            ]
        assert TestNamespaceContract.HEADERS, "no public headers found"
        return TestNamespaceContract.HEADERS

    def test_every_public_header_opens_namespace_orthon(self):
        for rel, src in self._headers():
            assert re.search(r"(?m)^namespace orthon \{$", src), (
                "%s does not open namespace orthon" % rel)
            assert re.search(r"(?m)^\}\s*//\s*namespace orthon$", src), (
                "%s does not close namespace orthon" % rel)

    def test_orthon_is_the_only_namespace_a_public_header_opens(self):
        for rel, src in self._headers():
            opened = re.findall(r"(?m)^namespace\s+(\w+)\s*\{", src)
            # Internal operator types live under orthon::dispatch. Keep
            # every other header's namespace rule and restrict the new nested
            # scopes to the canonical dispatch headers.
            allowed = {"orthon"}
            if rel in {"include/orthon/dispatch/operator.h",
                       "include/orthon/dispatch/dispatcher.h"}:
                allowed.add("dispatch")
            if rel in {"include/orthon/backends/cpu/kernels.h", "include/orthon/backends/cuda/kernels.h"}:
                allowed |= {"backend", "cpu" if "/cpu/" in rel else "cuda"}
            if rel == "include/orthon/ops/binary_ops.h":
                allowed.add("dispatch")
            if rel in {"include/orthon/distributed/work.h",
                       "include/orthon/distributed/process_group.h",
                       "include/orthon/distributed/nccl_process_group.h",
                       "include/orthon/distributed/nccl_transport.h"}:
                allowed.add("distributed")
            assert set(opened) == allowed, (
                "%s opens namespaces %s" % (rel, sorted(set(opened))))

    def test_no_public_declaration_sits_outside_the_namespace(self):
        """Anything above `namespace orthon {` must be preprocessor or comment."""
        for rel, src in self._headers():
            head = src[:src.index("namespace orthon {")]
            for line in _strip_c_comments(head).split("\n"):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if stripped.endswith("\\"):     # macro continuation
                    continue
                pytest.fail("%s declares %r outside namespace orthon"
                            % (rel, stripped))

    def test_generic_type_names_never_appear_at_global_scope(self):
        generic = ("Tensor", "TensorImpl", "Storage", "Node", "Device",
                   "DeviceType", "DeviceGuard", "DType", "Allocator")
        for rel, src in self._headers():
            head = _strip_c_comments(src[:src.index("namespace orthon {")])
            for name in generic:
                assert not re.search(r"\b(?:class|struct|enum class)\s+%s\b" % name, head), (
                    "%s declares %s at global scope" % (rel, name))

    def test_no_global_alias_reaches_into_the_namespace(self):
        """A global `using X = orthon::X;` would undo the namespace entirely."""
        for rel, src in _text_files():
            if not rel.endswith((".h", ".hpp", ".cuh", ".cpp", ".cu")):
                continue
            body = _strip_c_comments(src)
            assert not re.search(r"(?m)^\s*using\s+\w+\s*=\s*orthon::", body), (
                "%s defines a global alias into orthon::" % rel)

    def test_bindings_do_not_import_the_namespace_at_file_scope(self):
        """`_C` is the boundary; a file-scope using-directive blurs it."""
        for rel, _ in _walk("bindings"):
            if not rel.endswith(".cpp"):
                continue
            body = _strip_c_comments(_read(rel))
            for line in body.split("\n"):
                if line.startswith("using namespace orthon"):
                    pytest.fail("%s has a file-scope `using namespace orthon;`"
                                % rel)

    def test_pybind_module_stays_in_the_global_namespace(self):
        body = _strip_c_comments(_read("bindings/module.cpp"))
        assert "PYBIND11_MODULE(_C, m)" in body
        assert "namespace orthon {" not in body


# One name, everywhere
class TestProjectNaming:
    def test_cmake_declares_the_orthon_project_and_target(self):
        body = _read("CMakeLists.txt")
        assert re.search(r"(?m)^project\(Orthon\b", body)
        assert re.search(r"(?m)^add_library\(orthon ", body)
        assert "add_library(Orthon::orthon ALIAS orthon)" in body

    def test_distribution_metadata_is_orthon(self):
        body = _read("pyproject.toml")
        assert re.search(r'(?m)^name = "orthon"$', body)
        assert re.search(r'(?m)^wheel\.packages = \["python/orthon"\]$', body)

    def test_the_runtime_package_is_the_only_one(self):
        packages = sorted(
            name for name in os.listdir(os.path.join(REPO, "python"))
            if os.path.isdir(os.path.join(REPO, "python", name)))
        assert packages == ["orthon"], packages

    def test_the_include_root_is_the_only_one(self):
        roots = sorted(
            name for name in os.listdir(os.path.join(REPO, "include"))
            if os.path.isdir(os.path.join(REPO, "include", name)))
        assert roots == ["orthon"], roots

    def test_project_specific_build_identifiers_use_the_orthon_prefix(self):
        body = _read("CMakeLists.txt")
        for name in ("_ORTHON_WITH_CUDA", "_ORTHON_WITH_NCCL",
                     "_ORTHON_CUDA_ENABLED", "_ORTHON_NCCL_ENABLED",
                     "ORTHON_FETCH_GTEST", "ORTHON_GTEST_VERSION", "ORTHON_GTEST_COMMIT",
                     "ORTHON_CPP_TESTS", "orthon_add_cpp_test"):
            assert name in body, name
        # Generic user-facing build controls stay generic.
        for name in ("WITH_CUDA", "WITH_NCCL", "WITH_AVX2", "BUILD_TESTING"):
            assert name in body, name

    def test_every_framework_environment_variable_is_orthon_prefixed(self):
        """A framework variable that is not ORTHON_-prefixed squats on a name
        it does not own. Only the standard launcher and toolkit variables in
        EXTERNAL_ENV_VARS are exempt, because those contracts belong to
        someone else."""
        found = set()
        for rel, src in _text_files():
            found.update(re.findall(
                r"environ(?:\.get)?[\(\[]\s*[\"']([A-Z][A-Z0-9_]+)[\"']", src))
            found.update(re.findall(r"getenv\(\s*\"([A-Z][A-Z0-9_]+)\"", src))
        assert found, "expected the runtime to read at least one variable"
        unowned = sorted(name for name in found
                         if name not in EXTERNAL_ENV_VARS
                         and not name.startswith("ORTHON_"))
        assert unowned == [], unowned

    def test_public_headers_define_no_macros(self):
        """A macro in a header is the only kind a consumer can collide with.

        Orthon's headers define none at all - not even include guards, which
        are `#pragma once`. Every error-check macro lives inside the one
        translation unit that uses it, where its name cannot escape.
        """
        offenders = []
        for rel, _ in _walk("include"):
            if not rel.endswith((".h", ".hpp", ".cuh")):
                continue
            for name in re.findall(r"(?m)^\s*#define\s+(\w+)", _read(rel)):
                if not name.startswith("ORTHON_"):
                    offenders.append("%s defines %s" % (rel, name))
        assert offenders == [], offenders

    def test_the_named_error_check_macros_keep_the_orthon_prefix(self):
        """These are the macros that would be worth colliding with."""
        # The typed transport owns error translation rather than bindings; this
        # checker. Keep the naming rule for any remaining check macros and
        # prove that the transport checks genuine collective/communicator calls.
        for rel, src in _text_files():
            if rel.endswith((".cpp", ".cu")):
                for name in re.findall(r"(?m)^\s*#define\s+(\w*(?:CUDA|NCCL)\w*CHECK_OP)", src):
                    assert name.startswith("ORTHON_"), (rel, name)
        transport = _read("src/distributed/system_nccl_transport.cpp")
        assert "void checked(ncclResult_t result)" in transport
        for call in ("ncclCommInitRank", "ncclCommDestroy", "ncclCommAbort", "ncclCommGetAsyncError",
                     "ncclAllReduce", "ncclBroadcast", "ncclGetUniqueId"):
            assert re.search(r"checked\(\s*" + call + r"\(", transport), call


# Release metadata has one source of truth
def test_cmake_and_python_declare_the_same_version():
    """CMake carries the numeric base and the complete Python release spelling."""
    from packaging.version import Version

    cmake = re.search(r"(?m)^project\(Orthon\s+VERSION\s+([0-9][0-9.]*)",
                      _read("CMakeLists.txt"))
    assert cmake, "CMakeLists.txt does not declare project(Orthon VERSION ...)"
    python = re.search(r'(?m)^version = "([^"]+)"', _read("pyproject.toml"))
    assert python, "pyproject.toml does not declare a version"
    release = re.search(r'(?m)^set\(ORTHON_RELEASE_VERSION "([^"]+)"\)',
                        _read("CMakeLists.txt"))
    assert release, "CMakeLists.txt does not declare ORTHON_RELEASE_VERSION"
    assert release.group(1) == python.group(1)
    assert cmake.group(1) == Version(python.group(1)).base_version, (
        "CMake says %s, pyproject says %s" % (cmake.group(1), python.group(1)))


def test_legacy_pickle_rejection_message_is_actionable():
    """The refusal has to name the format it does read and what to do next."""
    body = _read("python/orthon/_serialization/archive.py")
    assert "legacy pickle checkpoints are unsupported" in body
    assert "trusted legacy build" in body


# Assets
class TestAssets:
    def test_the_project_mark_is_a_plain_svg(self):
        path = os.path.join(REPO, "assets", "orthon.svg")
        assert os.path.isfile(path), "assets/orthon.svg is missing"
        svg = _read("assets/orthon.svg")
        assert svg.lstrip().startswith("<svg")
        for banned in ("base64", "<image", "xlink:href", "@font-face",
                       "<script", "<foreignObject"):
            assert banned not in svg, "assets/orthon.svg contains %r" % banned
        # The SVG namespace declaration is the only permitted URL: nothing in
        # the mark may be fetched from anywhere.
        urls = re.findall(r"https?://[^\"'\s>]+", svg)
        assert urls == ["http://www.w3.org/2000/svg"], urls

    def test_no_raster_or_font_assets_are_shipped(self):
        assets = sorted(rel for rel, _ in _walk("assets"))
        assert assets == ["assets/orthon.svg"], assets

    def test_assets_carry_no_authoring_or_provenance_metadata(self):
        """The mark is plain geometry; nothing records what drew it."""
        markers = (b"content credentials", b"c2pa", b"xmp:", b"<rdf:",
                   b"<metadata", b"sodipodi", b"inkscape", b"illustrator",
                   b"exif")
        for rel, full in _walk("assets"):
            with open(full, "rb") as fh:
                blob = fh.read().lower()
            for marker in markers:
                assert marker not in blob, "%s carries %r" % (rel, marker)

    def test_the_readme_shows_the_mark_by_relative_path(self):
        readme = _read("README.md")
        assert 'src="assets/orthon.svg"' in readme


def _r4_stage_python(source, stage):
    import subprocess
    result = subprocess.run([
        'cmake', '-DORTHON_PYTHON_SOURCE_DIR=' + str(source),
        '-DORTHON_PYTHON_STAGE_DIR=' + str(stage),
        '-P', str(REPO / 'cmake/stage_python.cmake'),
    ], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_r4_python_stage_preserves_structure_and_excludes_artifacts(tmp_path):
    source, stage = tmp_path / 'source', tmp_path / 'stage'
    modules = {'__init__.py': 'from . import nn\n', 'nn/__init__.py': '# nn\n',
               'nn/linear.py': 'value = 1\n', '_random.py': 'value = 2\n'}
    artifacts = ['__pycache__/cached.py', '__pycache__/cached.pyc',
                 '.venv/unused.py', '.git/unused.py', 'build/unused.py',
                 'dist/unused.py', '_C.unwanted.so', 'notes.log']
    for name, content in {**modules, **dict.fromkeys(artifacts, 'artifact\n')}.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    stage.mkdir()
    extension = stage / '_C.sentinel.so'
    extension.write_bytes(b'compiled-extension-sentinel')
    _r4_stage_python(source, stage)
    assert {str(p.relative_to(stage)) for p in stage.rglob('*') if p.is_file()} == {
        *modules, extension.name,
    }
    assert all((stage / name).read_text() == content for name, content in modules.items())
    assert extension.read_bytes() == b'compiled-extension-sentinel'


def test_r4_python_stage_updates_edits_and_additions_without_touching_extension(tmp_path):
    source, stage = tmp_path / 'source', tmp_path / 'stage'
    source.mkdir()
    module = source / '__init__.py'
    module.write_text('value = 1\n')
    _r4_stage_python(source, stage)
    extension = stage / '_C.sentinel.so'
    extension.write_bytes(b'compiled-extension-sentinel')
    module.write_text('value = 2\n')
    os.utime(module, (1, 1))
    (source / 'added.py').write_text('value = 3\n')
    _r4_stage_python(source, stage)
    assert (stage / '__init__.py').read_text() == 'value = 2\n'
    assert (stage / 'added.py').read_text() == 'value = 3\n'
    assert extension.read_bytes() == b'compiled-extension-sentinel'


def test_r4_removed_python_source_requires_a_fresh_stage(tmp_path):
    source, stage = tmp_path / 'source', tmp_path / 'stage'
    source.mkdir()
    (source / '__init__.py').write_text('# package\n')
    removed = source / 'removed.py'
    removed.write_text('value = 1\n')
    _r4_stage_python(source, stage)
    removed.unlink()
    _r4_stage_python(source, stage)
    assert (stage / 'removed.py').exists()
    fresh = tmp_path / 'fresh-stage'
    _r4_stage_python(source, fresh)
    assert (fresh / '__init__.py').exists() and not (fresh / 'removed.py').exists()
