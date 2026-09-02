import sys
import os
import importlib.util

# When orthon hasn't been installed (e.g. no `pip install -e .`), fall back
# to the plain-cmake build directory so tests still work with PYTHONPATH=build.
if importlib.util.find_spec("orthon") is None:
    _build = os.path.join(os.path.dirname(__file__), "..", "..", "build")
    if os.path.isdir(_build):
        sys.path.insert(0, _build)
