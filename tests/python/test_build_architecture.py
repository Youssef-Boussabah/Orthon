"""Orthon's build and source-layout invariants, audited at source level.

Four things this file keeps true, none of which any behavioural test can see:

1. The CMake source lists are EXPLICIT and COMPLETE. A glob would make a stale
   build tree compile a different set of files than the tree on disk, and would
   let a dead source stay in the library because nothing ever names it.
2. The public C++ API lives in ``namespace orthon``. Generic names like
   ``Tensor``, ``Storage``, ``Device`` and ``Node`` must never sit in the
   global namespace of a header a consumer includes.
3. The project speaks with one name: ``orthon`` / ``Orthon`` / ``ORTHON_`` in
   packages, include roots, build identifiers and environment variables.
4. No file is a meaningless empty placeholder, and release metadata has a
   single source of truth.

The native counterpart is tests/cpp/test_identity.cpp, which proves the same
namespace contract by COMPILING against it.
"""
import os
import re

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

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

# Environment variables the runtime reads that are NOT Orthon's to name: the
# standard distributed-launcher contract and the CUDA toolkit's own.
EXTERNAL_ENV_VARS = {"RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
                     "MASTER_ADDR", "MASTER_PORT", "CUDA_PATH",
                     "CUDA_VISIBLE_DEVICES"}


def _read(rel):
    with open(os.path.join(REPO, rel), encoding="utf-8") as fh:
        return fh.read()


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


def _strip_c_comments(src):
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return re.sub(r"//[^\n]*", "", src)


def _cmake_source():
    """The root CMakeLists.txt with comments stripped."""
    return re.sub(r"(?m)#[^\n]*", "", _read("CMakeLists.txt"))


def _cmake_list(name):
    """The entries of a `set(<name> ...)` block in the root CMakeLists.txt."""
    match = re.search(r"set\(\s*" + name + r"\s(.*?)\n\s*\)", _cmake_source(), re.S)
    assert match, "CMakeLists.txt has no set(%s ...) block" % name
    return sorted(match.group(1).split())


# ---------------------------------------------------------------------------
# Explicit CMake source lists
# ---------------------------------------------------------------------------

class TestSourceLists:
    def test_no_source_globs_remain(self):
        """A glob does not re-run configure when a file is added or deleted."""
        body = _cmake_source()
        assert not re.search(r"(?i)\bfile\s*\(\s*GLOB", body), (
            "CMakeLists.txt must list sources explicitly, never glob them")

    def test_core_source_list_is_exactly_the_live_cpp_tree(self):
        listed = _cmake_list("ORTHON_CORE_SOURCES")
        on_disk = sorted(rel for rel, _ in _walk("src") if rel.endswith(".cpp"))
        assert listed == on_disk, (
            "ORTHON_CORE_SOURCES disagrees with src/**.cpp\n"
            "  listed not on disk: %s\n"
            "  on disk not listed: %s"
            % (sorted(set(listed) - set(on_disk)),
               sorted(set(on_disk) - set(listed))))

    def test_cuda_source_list_is_exactly_the_live_cu_tree(self):
        listed = _cmake_list("ORTHON_CUDA_SOURCES")
        on_disk = sorted(rel for rel, _ in _walk("src") if rel.endswith(".cu"))
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
        listed = _cmake_list("ORTHON_CPP_TESTS")
        on_disk = sorted(
            os.path.splitext(os.path.basename(rel))[0]
            for rel, _ in _walk("tests/cpp") if rel.endswith(".cpp"))
        assert listed == on_disk


# ---------------------------------------------------------------------------
# No meaningless empty files
# ---------------------------------------------------------------------------

def test_no_zero_byte_source_or_header_files():
    empty = [rel for rel, full in _walk()
             if os.path.getsize(full) == 0
             and os.path.splitext(rel)[1] in
             (".py", ".h", ".hpp", ".cuh", ".cpp", ".cu", ".md", ".toml",
              ".json", ".yml", ".yaml", ".cmake", ".txt", ".svg")]
    assert empty == [], "zero-byte placeholder files: %s" % empty


# ---------------------------------------------------------------------------
# C++ namespace contract
# ---------------------------------------------------------------------------

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
                if rel.endswith((".h", ".hpp", ".cuh"))
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
            assert set(opened) == {"orthon"}, (
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


# ---------------------------------------------------------------------------
# One name, everywhere
# ---------------------------------------------------------------------------

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
        defined = set()
        for rel, src in _text_files():
            if rel.endswith((".cpp", ".cu")):
                defined.update(
                    re.findall(r"(?m)^\s*#define\s+(ORTHON_\w+)", src))
        assert {"ORTHON_CUDA_CHECK", "ORTHON_CUDA_CHECK_OP",
                "ORTHON_NCCL_CHECK_OP", "ORTHON_INDEX_CUDA_CHECK",
                "ORTHON_SLICE_CUDA_CHECK"} <= defined, sorted(defined)


# ---------------------------------------------------------------------------
# Release metadata has one source of truth
# ---------------------------------------------------------------------------

def test_cmake_and_python_declare_the_same_version():
    """Two version literals that can drift are two chances to ship a wrong one."""
    cmake = re.search(r"(?m)^project\(Orthon\s+VERSION\s+([0-9][0-9.]*)",
                      _read("CMakeLists.txt"))
    assert cmake, "CMakeLists.txt does not declare project(Orthon VERSION ...)"
    python = re.search(r'(?m)^version = "([^"]+)"', _read("pyproject.toml"))
    assert python, "pyproject.toml does not declare a version"
    assert cmake.group(1) == python.group(1), (
        "CMake says %s, pyproject says %s" % (cmake.group(1), python.group(1)))


def test_legacy_pickle_rejection_message_is_actionable():
    """The refusal has to name the format it does read and what to do next."""
    body = _read("python/orthon/checkpoint.py")
    assert "legacy pickle checkpoints are unsupported" in body
    assert "trusted legacy build" in body


# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------

class TestAssets:
    def test_the_project_mark_is_a_plain_svg(self):
        path = os.path.join(REPO, "assets", "orthon.svg")
        assert os.path.isfile(path), "assets/orthon.svg is missing"
        svg = _read("assets/orthon.svg")
        assert svg.lstrip().startswith("<svg")
        for banned in ("base64", "<image", "xlink:href", "@font-face",
                       "<script", "<foreignObject"):
            assert banned not in svg, "assets/orthon.svg contains %r" % banned
        # The SVG namespace declaration is the ONLY permitted URL: nothing in
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
