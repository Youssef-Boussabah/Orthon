"""Local Git inventory and explicit repository hygiene boundaries."""
from pathlib import PurePosixPath
import subprocess
import re

from .source import REPO_ROOT


TEXT_SUFFIXES = frozenset({
    ".py", ".cpp", ".cu", ".h", ".hpp", ".cuh", ".md", ".toml", ".json",
    ".yml", ".yaml", ".svg", ".cmake", ".txt",
})
TEXT_NAMES = frozenset({
    ".gitattributes", ".gitignore", ".clang-format", "CMakeLists.txt", "LICENSE",
})
GENERATED_DIRS = frozenset({
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".cache",
    "CMakeFiles", "build", "dist", "_skbuild", ".venv", "venv", "htmlcov",
})
GENERATED_SUFFIXES = frozenset({
    ".pyc", ".pyo", ".pyd", ".o", ".obj", ".a", ".lib", ".so", ".dll",
    ".dylib", ".exe", ".ncu-rep", ".nsys-rep", ".qdrep", ".swp", ".swo",
    ".cubin", ".ptx", ".fatbin", ".ckpt", ".orthon", ".log", ".xml", ".zip",
    ".whl", ".sha256",
})
GENERATED_NAMES = frozenset({
    "uv.lock", "CMakeCache.txt", "cmake_install.cmake", "CTestTestfile.cmake",
    "compile_commands.json", ".coverage", "build.ninja",
})


def git(*args, input=None):
    result = subprocess.run(["git", *args], cwd=REPO_ROOT, input=input,
                            capture_output=True, check=True)
    return result.stdout


def tracked_entries():
    entries = {}
    for record in git("ls-files", "--stage", "-z").split(b"\0"):
        if record:
            header, path = record.split(b"\t", 1)
            mode, _, stage = header.decode().split()
            assert stage == "0", "unmerged repository entry: " + path.decode()
            entries[path.decode()] = mode
    assert entries, "repository architecture requires local Git metadata"
    return entries


def repository_paths():
    """Tracked files plus visible working-tree additions."""
    added = git("ls-files", "--others", "--exclude-standard", "-z")
    return sorted({p for p in tracked_entries() if (REPO_ROOT / p).exists()} | {
        p.decode() for p in added.split(b"\0") if p
    })


def is_text(path):
    path = PurePosixPath(path)
    return path.suffix in TEXT_SUFFIXES or path.name in TEXT_NAMES


def is_generated(path):
    if path == "examples/results/.gitkeep":
        return False
    path = PurePosixPath(path)
    return (bool(set(path.parts) & GENERATED_DIRS)
            or any(p.startswith(("build_", "cmake-build-")) for p in path.parts)
            or path.suffix in GENERATED_SUFFIXES
            or path.name in GENERATED_NAMES
            or path.name.startswith(".ninja_")
            or path.name.endswith("~")
            or (path.suffix == ".sqlite" and "profiles" in path.parts)
            or path.parent == PurePosixPath("examples/results"))


def is_review_artifact(path):
    path = PurePosixPath(path)
    return (bool(set(path.parts) & {"reviews", "archive", "evidence", "reports", "profiles"})
            or bool(re.match(r"(?:v\d+_|legacy_deletion_gate|ddp_cloud_validation)", path.name))
            or path.name.upper().startswith(("HUMANIZATION_REPORT", "AUDIT_REPORT")))
