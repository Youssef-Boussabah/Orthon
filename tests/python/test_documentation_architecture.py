"""Current documentation boundaries, references and public inventory."""
import ast
import importlib
import re

from tests.support.documentation import (
    CURRENT_DOCUMENTS, PUBLIC_NAMES, RETIRED_TOKENS, STALE_STREAM_PHRASES,
    api_manifest, command_paths, fences, internal_links, source_paths,
)
from tests.support.source import REPO_ROOT


def current_texts():
    return {name: (REPO_ROOT / name).read_text(encoding="utf-8")
            for name in CURRENT_DOCUMENTS}


def test_reviewed_current_manifest_and_index():
    assert len(CURRENT_DOCUMENTS) == len(set(CURRENT_DOCUMENTS))
    assert all((REPO_ROOT / name).is_file() for name in CURRENT_DOCUMENTS)
    actual = {"README.md"} | {p.relative_to(REPO_ROOT).as_posix()
                              for p in (REPO_ROOT / "docs").glob("*.md")}
    assert set(CURRENT_DOCUMENTS) == actual
    index = (REPO_ROOT / "docs/README.md").read_text()
    targets = {(REPO_ROOT / "docs" / p).resolve() for p in internal_links(index)}
    assert {(REPO_ROOT / p).resolve() for p in CURRENT_DOCUMENTS
            if p != "docs/README.md"} <= targets


def test_current_internal_markdown_links_exist():
    missing = [(name, target) for name, text in current_texts().items()
               for target in internal_links(text)
               if not (REPO_ROOT / name).parent.joinpath(target).exists()]
    assert not missing, missing


def test_current_explicit_repository_paths_exist():
    missing = [(name, path) for name, text in current_texts().items()
               for path in source_paths(text) if not (REPO_ROOT / path).exists()]
    assert not missing, missing


def test_current_command_modules_and_scripts_exist():
    missing = [(name, path) for name, text in current_texts().items()
               for path in command_paths(text) if not (REPO_ROOT / path).is_file()]
    assert not missing, missing


def test_current_docs_exclude_retired_surfaces():
    found = [(name, token) for name, text in current_texts().items()
             for token in RETIRED_TOKENS if token in text]
    assert not found, found


def test_current_docs_exclude_known_stale_stream_claims():
    found = [(name, phrase) for name, text in current_texts().items()
             for phrase in STALE_STREAM_PHRASES if phrase in text.lower()]
    assert not found, found


def test_current_docs_exclude_local_machine_paths():
    found = [(name, value) for name, text in current_texts().items()
             for value in re.findall(r"/(?:home|Users)/[\w.-]+/\S*|[A-Za-z]:\\Users\\\S*", text)]
    assert not found, found


def test_public_api_manifest_matches_reviewed_scope_and_package():
    documented = api_manifest((REPO_ROOT / "docs/api.md").read_text())
    assert documented == PUBLIC_NAMES
    for module, names in documented.items():
        loaded = importlib.import_module(module)
        assert all(hasattr(loaded, name) for name in names), module
        if hasattr(loaded, "__all__"):
            assert {n for n in loaded.__all__ if not n.startswith("_")} == names, module
        assert all(not name.startswith("_") for name in names), module

    # The Python setter discards the native C++ previous-state return value.
    orthon = importlib.import_module("orthon")
    previous = orthon.is_grad_enabled()
    assert isinstance(previous, bool)
    try:
        assert orthon.set_grad_enabled(previous) is None
    finally:
        orthon.set_grad_enabled(previous)


def test_public_api_entries_are_explained_in_reference():
    text = (REPO_ROOT / "docs/api.md").read_text()
    prose = re.sub(r"^```json\s*\n.*?^```\s*$", "", text, flags=re.M | re.S)
    for names in PUBLIC_NAMES.values():
        assert all(re.search(r"\b" + re.escape(name) + r"\b", prose) for name in names)


def test_current_python_examples_parse():
    for name, text in current_texts().items():
        for index, code in enumerate(fences(text, "python")):
            ast.parse(code, filename=f"{name}:python-block-{index + 1}")


def test_reference_parsers_distinguish_examples_urls_and_repository_targets():
    text = '''[source](../src/core/buffer.cpp#owner)
[web](https://example.invalid/missing) [anchor](#owner)
[encoded](<a%20file.md>)
[ref]: ../README.md
```python
signature = "[argument](missing.md)"
```
`src/core/buffer.cpp` `src/**/*.cpp` `Tensor::backward(bool)`
python -m benchmarks.compare_pytorch --help
python examples/ddp_local_demo.py
'''
    assert list(internal_links(text)) == ["../src/core/buffer.cpp", "a file.md", "../README.md"]
    assert list(source_paths(text)) == ["src/core/buffer.cpp"]
    assert list(command_paths(text)) == ["benchmarks/compare_pytorch.py", "examples/ddp_local_demo.py"]
