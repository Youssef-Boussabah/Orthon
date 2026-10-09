"""Repository identity, packaging metadata and local hygiene fences."""
import io
import hashlib
import os
import re
import subprocess
import sys
import tarfile
import textwrap
import zipfile

try:
    import tomllib
except ModuleNotFoundError:  # pytest supplies tomli on supported Python 3.10.
    import tomli as tomllib

from packaging.requirements import Requirement
from packaging.version import Version

from tests.support.documentation import CURRENT_DOCUMENTS, internal_links
from tests.support.repository import (
    git, is_generated, is_review_artifact, is_text, repository_paths, tracked_entries,
)
from tests.support.source import REPO_ROOT, read_text


def test_required_repository_files_exist():
    required = {
        "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "CONTRIBUTING.md",
        ".gitattributes", ".gitignore", "pyproject.toml", "CMakeLists.txt",
        "CMakePresets.json",
    }
    assert all((REPO_ROOT / p).is_file() for p in required)


def test_tracked_paths_exclude_generated_artifacts():
    assert not [p for p in repository_paths()
                if is_generated(p) or is_review_artifact(p)]


def test_tracked_regular_files_are_non_executable():
    entries = tracked_entries()
    assert not {p: mode for p, mode in entries.items()
                if mode.startswith("100") and mode != "100644"}
    if os.name == "posix":
        assert not [p for p in repository_paths()
                    if (REPO_ROOT / p).is_file()
                    and not (REPO_ROOT / p).is_symlink()
                    and (REPO_ROOT / p).stat().st_mode & 0o111]


def test_tracked_text_encoding_and_line_endings():
    for p in repository_paths():
        if is_text(p):
            data = (REPO_ROOT / p).read_bytes()
            data.decode("utf-8")
            assert not data.startswith(b"\xef\xbb\xbf"), p
            assert b"\r\n" not in data, p
    assert "* text=auto eol=lf" in read_text(".gitattributes").splitlines()


def test_ignore_policy_preserves_source_and_excludes_local_artifacts():
    rules = {line.strip() for line in read_text(".gitignore").splitlines()
             if line.strip() and not line.lstrip().startswith("#")}
    assert "uv.lock" in rules or "/uv.lock" in rules
    assert "*.cmake" not in rules
    ignored = [
        "uv.lock", "build/CMakeCache.txt", "CMakeFiles/example", "dist/a.whl",
        ".ninja_log", "__pycache__/a.pyc", ".pytest_cache/a", ".mypy_cache/a",
        ".ruff_cache/a", ".cache/a", ".coverage", "htmlcov/index.html",
        ".venv/example", ".env", "profiles/a.nsys-rep", "profiles/a.sqlite",
        "examples/results/checkpoint.orthon", "data/example", "pip-install-x/a",
    ]
    output = git("check-ignore", "--no-index", "--stdin",
                 input=("\n".join(ignored) + "\n").encode()).decode().splitlines()
    assert set(output) == set(ignored)
    visible = ["cmake/future-module.cmake", "examples/results/.gitkeep",
               "assets/orthon.svg", "CONTRIBUTING.md"]
    result = subprocess.run(["git", "check-ignore", "--no-index", "--stdin"],
                            cwd=REPO_ROOT, input="\n".join(visible) + "\n",
                            text=True, capture_output=True)
    assert result.returncode == 1 and not result.stdout, result


def test_package_metadata_matches_maintained_project():
    data = tomllib.loads(read_text("pyproject.toml"))
    project = data["project"]
    assert project["name"] == "orthon"
    # Existing package/build identity cases own the exact version and agreement.
    assert re.search(r"project\(Orthon VERSION " + re.escape(Version(project["version"]).base_version)
                     + r"\b", read_text("CMakeLists.txt"))
    assert project["license"] == "MIT"
    assert set(project["license-files"]) == {"LICENSE", "THIRD_PARTY_NOTICES.md"}
    assert {"name": "Youssef Boussabah"} in project["maintainers"]
    assert "authors" not in project and "urls" not in project
    assert project["requires-python"] == ">=3.10"
    requirements = {r.name: r for r in map(Requirement, data["build-system"]["requires"])}
    backend = requirements["scikit-build-core"]
    floors = [Version(s.version) for s in backend.specifier if s.operator in (">=", ">")]
    assert floors and max(floors) >= Version("0.11")
    assert data["tool"]["scikit-build"]["minimum-version"] == "build-system.requires"
    assert not any(c.startswith("License ::") for c in project.get("classifiers", []))


def test_pybind11_notice_preserves_authenticated_upstream_license():
    notice = read_text("THIRD_PARTY_NOTICES.md")
    assert "## pybind11" in notice
    blocks = re.findall(r"^```text\n(.*?)^```$", notice, re.M | re.S)
    assert len(blocks) == 1
    # SHA-256 of the complete LICENSE from the pybind11 3.1.0 build dependency.
    assert hashlib.sha256(blocks[0].encode()).hexdigest() == (
        "83965b843b98f670d3a85bd041ed4b372c8ec50d7b4a5995a83ac697ba675dcb")


def test_current_project_identifiers_preserve_wire_contract_and_portability():
    paths = [p for p in repository_paths() if is_text(p)]
    assert paths
    for p in paths:
        text = read_text(p).replace("ml-runtime-checkpoint", "")
        assert not re.search(r"/(?:home|Users)/[\w.-]+/|[A-Za-z]:\\Users\\", text), p
        if p.startswith("tests/"):
            continue
        banned = ["TensorForge", "ml-runtime-remodel", "ml-runtime"]
        assert not [name for name in banned if name.lower() in text.lower()], p


def test_ci_uses_durable_hardware_and_version_boundaries():
    text = read_text(".github/workflows/ci.yml")
    assert "orthon-0." not in text
    assert "docs/testing.md" in text
    assert "runs-on: ubuntu-latest" in text


def test_ci_artifact_validation_accepts_future_project_versions(tmp_path):
    text = read_text(".github/workflows/ci.yml")
    step = text.split("- name: Assert the artifacts are named and laid out as orthon", 1)[1]
    script = textwrap.dedent(step.split("python - <<'PY'\n", 1)[1].split("          PY\n", 1)[0])
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "orthon"\nversion = "2.0.0"\nlicense = "MIT"\n')
    license_bytes = (REPO_ROOT / "LICENSE").read_bytes()
    (tmp_path / "LICENSE").write_bytes(license_bytes)
    dist = tmp_path / "dist"
    dist.mkdir()
    metadata = ("Metadata-Version: 2.4\nName: orthon\nVersion: 2.0.0\n"
                "License-Expression: MIT\nLicense-File: LICENSE\n\n").encode()
    with zipfile.ZipFile(dist / "orthon-2.0.0-py3-none-any.whl", "w") as wheel:
        wheel.writestr("orthon/__init__.py", "")
        wheel.writestr("orthon-2.0.0.dist-info/METADATA", metadata)
        wheel.writestr("orthon-2.0.0.dist-info/licenses/LICENSE", license_bytes)
    with tarfile.open(dist / "orthon-2.0.0.tar.gz", "w:gz") as sdist:
        for name, data in [("PKG-INFO", metadata), ("LICENSE", license_bytes)]:
            info = tarfile.TarInfo("orthon-2.0.0/" + name)
            info.size = len(data)
            sdist.addfile(info, io.BytesIO(data))
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    # The same validator must reject metadata that disagrees with the project.
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "orthon"\nversion = "2.0.1"\nlicense = "MIT"\n')
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode != 0


def test_ci_contains_no_release_or_publish_job():
    text = read_text(".github/workflows/ci.yml").lower()
    for action in ["pypi publish", "gh release", "twine upload", "release upload",
                   "git push", "pypa/gh-action-pypi-publish", "actions/upload-artifact"]:
        assert action not in text


def test_repository_entry_points_link_metadata():
    expected = {(REPO_ROOT / p).resolve()
                for p in ["CONTRIBUTING.md", "LICENSE"]}
    for p in ["README.md", "docs/README.md"]:
        links = {(REPO_ROOT / p).parent.joinpath(target).resolve()
                 for target in internal_links(read_text(p))}
        assert expected <= links
    assert all(REPO_ROOT.joinpath(target).exists()
               for target in internal_links(read_text("CONTRIBUTING.md")))
