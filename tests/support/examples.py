"""Execute repository examples with a scoped sibling-import path."""
import importlib.util
import sys
from .source import REPO_ROOT

def load_example(filename):
    directory = REPO_ROOT / "examples"
    path = directory / filename
    if path.parent != directory or path.suffix != ".py":
        raise ValueError("expected an example Python filename")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    original_path = sys.path[:]
    try:
        sys.path.insert(0, str(directory))
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_path
    return module
