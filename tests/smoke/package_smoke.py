"""Runtime smoke test for an installed orthon wheel.

Run against a CLEAN environment that has the wheel and nothing else:

    python -m venv /tmp/smoke && /tmp/smoke/bin/pip install dist/*.whl
    /tmp/smoke/bin/python tests/smoke/package_smoke.py

This is the same script the `package` job in .github/workflows/ci.yml runs, so
CI and a local check exercise identical code rather than two drifting copies.

What it is for - packaging correctness, pinned under automation:

  * `import orthon` succeeds from a wheel install alone;
  * the declared runtime dependency set is sufficient - NumPy is pulled in
    transitively, nothing else is needed;
  * torch and pytest are TEST-only dependencies and must not be required to
    import or use the package. They are declared under
    [project.optional-dependencies].test precisely so a normal install does not
    drag a multi-gigabyte reference framework along.

It deliberately uses no test framework: pytest is one of the things being
proven unnecessary.
"""
import sys


def fail(message):
    raise SystemExit("package smoke FAILED: " + message)


# The environment must be clean, or this script proves less than it claims.
# pickle is NOT on this list: NumPy imports it, so its presence in sys.modules
# says nothing about orthon. The checkpoint section below proves the useful
# property instead, by making every pickle entry point fatal.
for forbidden in ("torch", "pytest"):
    if forbidden in sys.modules:
        fail(f"{forbidden} was already imported; this is not a clean environment")

import orthon  # noqa: E402

print("orthon     :", orthon.__file__)
print("python       :", sys.version.split()[0])
print("cuda_enabled :", orthon.cuda_enabled())

# Importing the package must not drag in the reference stack.
if "torch" in sys.modules:
    fail("importing orthon pulled in torch")
if "pytest" in sys.modules:
    fail("importing orthon pulled in pytest")

# ---- create tensors ----
a = orthon.full([2, 3], 2.0)
b = orthon.full([2, 3], 5.0)
if tuple(a.shape) != (2, 3):
    fail(f"unexpected shape {a.shape}")
if a.dtype != "float32":
    fail(f"unexpected dtype {a.dtype}")

# ---- basic add / mul ----
# Tensor.tolist() is flat by contract, so compare against flat expectations.
added = orthon.add(a, b).tolist()
if added != [7.0] * 6:
    fail(f"add produced {added}")

multiplied = orthon.mul(a, b).tolist()
if multiplied != [10.0] * 6:
    fail(f"mul produced {multiplied}")

# ---- sub / div and the Python scalar operators ----
subtracted = orthon.sub(b, a).tolist()
if subtracted != [3.0] * 6:
    fail(f"sub produced {subtracted}")

divided = orthon.div(b, a).tolist()
if divided != [2.5] * 6:
    fail(f"div produced {divided}")

for label, got, expected in (
    ("a + 1",   (a + 1).tolist(),   [3.0] * 6),
    ("1 + a",   (1 + a).tolist(),   [3.0] * 6),
    ("a - 0.5", (a - 0.5).tolist(), [1.5] * 6),
    ("3 - a",   (3 - a).tolist(),   [1.0] * 6),
    ("a * 3",   (a * 3).tolist(),   [6.0] * 6),
    ("3 * a",   (3 * a).tolist(),   [6.0] * 6),
    ("a / 4",   (a / 4).tolist(),   [0.5] * 6),
    ("4 / a",   (4 / a).tolist(),   [2.0] * 6),
):
    if got != expected:
        fail(f"{label} produced {got}, expected {expected}")

# A scalar operand must not materialise a data-shaped temporary.
if tuple(orthon._scalar_like(a, 2.0).shape) != ():
    fail("scalar operand is not rank-0")

# ---- simple backward ----
x = orthon.full([3], 3.0)
x.requires_grad_(True)
w = orthon.full([3], 4.0)
orthon.mul(x, w).sum().backward()
if x.grad is None:
    fail("backward produced no gradient")
if x.grad.tolist() != [4.0, 4.0, 4.0]:
    fail(f"d(x*w)/dx produced {x.grad.tolist()}, expected w")

# ---- a composed graph, so the smoke covers more than one node ----
y = orthon.full([4], 1.0)
y.requires_grad_(True)
negative = orthon.mul(y, orthon.full([4], -1.0))
orthon.relu(negative).sum().backward()
if y.grad.tolist() != [0.0] * 4:
    fail(f"relu(-y) gradient was {y.grad.tolist()}, expected all zeros")

# ---- checkpoint file APIs ----
#
# Worth proving from a clean install specifically because the format is
# stdlib-only: zipfile + json + hashlib, no pickle, and no NumPy needed to
# reconstruct a tensor.
import json  # noqa: E402
import os  # noqa: E402
import pickle  # noqa: E402
import tempfile  # noqa: E402
import zipfile  # noqa: E402

from orthon.checkpoint import FORMAT_ID, FORMAT_VERSION  # noqa: E402


def _never_unpickle(*args, **kwargs):
    fail("a checkpoint API called into pickle")


# Armed for the whole checkpoint section below. Asserting that `pickle` is
# absent from sys.modules would prove nothing - NumPy imports it - so this
# proves the stronger and more useful thing directly: nothing orthon does
# here reaches the pickle VM.
pickle.load = _never_unpickle
pickle.loads = _never_unpickle
pickle.dump = _never_unpickle
pickle.dumps = _never_unpickle


class _TinyModel:
    """No nn.Module needed: the file APIs only require the state_dict pair."""

    def __init__(self, tensor):
        self.tensor = tensor

    def state_dict(self):
        return {"w": self.tensor}

    def load_state_dict(self, state, strict=True):
        self.tensor = state["w"]


with tempfile.TemporaryDirectory() as _tmp:
    _path = os.path.join(_tmp, "smoke.ckpt")
    _source = orthon.full([3, 2], 0.375)
    _payload = orthon._tensor_to_bytes(_source)
    orthon.save_checkpoint(_path, model=_TinyModel(_source),
                             metadata={"epoch": 2, "tag": "smoke"})

    # It is a ZIP with a JSON manifest, readable without executing anything.
    with open(_path, "rb") as _handle:
        if _handle.read(4) != b"PK\x03\x04":
            fail("checkpoint is not a ZIP archive")
    with zipfile.ZipFile(_path) as _archive:
        _manifest = json.loads(_archive.read("manifest.json"))
        if _manifest["format"] != FORMAT_ID:
            fail(f"unexpected format id {_manifest['format']!r}")
        if _manifest["format_version"] != FORMAT_VERSION:
            fail(f"unexpected format version {_manifest['format_version']!r}")
        if _archive.read("tensors/000000.bin") != _payload:
            fail("tensor payload is not the raw tensor bytes")

    _destination = _TinyModel(orthon.zeros([3, 2]))
    _metadata = orthon.load_checkpoint(_path, model=_destination)
    if _metadata != {"epoch": 2, "tag": "smoke"}:
        fail(f"metadata round-trip produced {_metadata}")
    if orthon._tensor_to_bytes(_destination.tensor) != _payload:
        fail("tensor did not round-trip bit-for-bit")

    # state-only API, and a legacy pickle must be refused rather than loaded.
    _state_path = os.path.join(_tmp, "state.ckpt")
    orthon.save_state_dict(_TinyModel(_source), _state_path)
    _reloaded = _TinyModel(orthon.zeros([3, 2]))
    orthon.load_state_dict(_reloaded, _state_path)
    if orthon._tensor_to_bytes(_reloaded.tensor) != _payload:
        fail("save_state_dict/load_state_dict did not round-trip")

    _legacy = os.path.join(_tmp, "legacy.bin")
    with open(_legacy, "wb") as _handle:
        _handle.write(b"\x80\x05not-a-real-pickle")
    try:
        orthon.load_checkpoint(_legacy)
    except RuntimeError as _exc:
        if "legacy pickle" not in str(_exc):
            fail(f"unexpected legacy rejection message: {_exc}")
    else:
        fail("a legacy pickle file was not rejected")

print("checkpoint  :", FORMAT_ID, "v%d" % FORMAT_VERSION)
print("SMOKE OK")
