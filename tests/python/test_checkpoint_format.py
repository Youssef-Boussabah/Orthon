"""Raw-byte checkpoint integrity, determinism, atomicity and safe decoding.

Behavioral model/optimizer round trips live in test_checkpoint.py."""
from tests.support.capabilities import orthon_cuda_available
import hashlib
import io
import json
import os
import pickle
import re
import zipfile

import numpy as np
import pytest

import orthon as ortho
import orthon.nn as nn
from orthon.checkpoint import FORMAT_ID, FORMAT_VERSION

CUDA = orthon_cuda_available()
requires_cuda = pytest.mark.skipif(not CUDA, reason="CUDA not available")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Helpers
class Holder:
    """Minimal state_dict/load_state_dict duck type for single-tensor cases."""

    def __init__(self, tensor):
        self.tensor = tensor

    def state_dict(self):
        return {"t": self.tensor}

    def load_state_dict(self, state, strict=True):
        incoming = state["t"]
        with ortho.no_grad():
            moved = incoming.to(self.tensor.device)
        self.tensor = moved


def _model():
    return nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))


def _raw(tensor):
    with ortho.no_grad():
        return ortho._tensor_to_bytes(tensor.to("cpu"))


def _from_numpy(array):
    return ortho.from_numpy(np.ascontiguousarray(array))


def _manifest(path):
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("manifest.json").decode("utf-8"))


def _rewrite(source, target, mutate_manifest=None, mutate_entries=None):
    """Copy an archive, optionally corrupting the manifest or the payloads."""
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        blobs = {name: archive.read(name) for name in names}
    manifest = json.loads(blobs["manifest.json"].decode("utf-8"))
    if mutate_manifest is not None:
        manifest = mutate_manifest(manifest)
    if manifest is not None:
        blobs["manifest.json"] = json.dumps(manifest).encode("utf-8")
    else:
        del blobs["manifest.json"]
        names = [n for n in names if n != "manifest.json"]
    if mutate_entries is not None:
        blobs, names = mutate_entries(blobs, names)
    with zipfile.ZipFile(target, "w") as archive:
        for name in names:
            archive.writestr(name, blobs[name])
    return target


# the raw tensor byte codec
DTYPE_CASES = [
    ("float32", np.array([1.0, -0.0, 0.0, np.inf, -np.inf, np.nan, 1 / 3,
                          3.4028235e38], dtype=np.float32)),
    ("float64", np.array([1.0, -0.0, np.inf, -np.inf, np.nan, 1 / 3,
                          1.7976931348623157e308], dtype=np.float64)),
    ("int32", np.array([-2147483648, -1, 0, 1, 2147483647], dtype=np.int32)),
    ("int64", np.array([-9223372036854775808, -1, 0, 1,
                        9223372036854775807], dtype=np.int64)),
    ("uint8", np.array([0, 1, 127, 255], dtype=np.uint8)),
    ("int8", np.array([-128, -1, 0, 1, 127], dtype=np.int8)),
    ("bool", np.array([True, False, True, True], dtype=np.bool_)),
]


@pytest.mark.parametrize("dtype,array", DTYPE_CASES)
def test_tensor_bytes_match_the_platform_representation(dtype, array):
    tensor = _from_numpy(array)
    assert tensor.dtype == dtype
    assert ortho._tensor_to_bytes(tensor) == array.tobytes()


@pytest.mark.parametrize("dtype,array", DTYPE_CASES)
def test_tensor_bytes_round_trip(dtype, array):
    tensor = _from_numpy(array)
    payload = ortho._tensor_to_bytes(tensor)
    restored = ortho._tensor_from_bytes(list(tensor.shape), dtype, payload)
    assert restored.dtype == dtype
    assert tuple(restored.shape) == tuple(tensor.shape)
    assert ortho._tensor_to_bytes(restored) == payload


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_low_precision_bytes_round_trip(dtype):
    source = _from_numpy(np.array([1e-5, 65504.0, 1 / 3, -0.0], dtype=np.float32))
    tensor = ortho.cast(source, dtype)
    payload = ortho._tensor_to_bytes(tensor)
    assert len(payload) == 8                      # 4 elements, 2 bytes each
    restored = ortho._tensor_from_bytes([4], dtype, payload)
    assert ortho._tensor_to_bytes(restored) == payload


def test_restored_tensor_is_a_plain_cpu_leaf():
    restored = ortho._tensor_from_bytes([2], "float32", b"\x00" * 8)
    assert restored.device.is_cpu()
    assert restored.is_contiguous
    assert restored.requires_grad is False
    assert restored._has_grad_fn is False
    assert restored.is_leaf is True


def test_empty_and_rank0_tensors():
    assert ortho._tensor_to_bytes(ortho.zeros([0])) == b""
    assert ortho._tensor_to_bytes(ortho.zeros([3, 0, 2])) == b""
    assert len(ortho._tensor_to_bytes(ortho.full([], 1.0))) == 4

    empty = ortho._tensor_from_bytes([0], "float32", b"")
    assert int(empty.numel) == 0
    zero_dim = ortho._tensor_from_bytes([3, 0, 2], "float64", b"")
    assert tuple(zero_dim.shape) == (3, 0, 2)
    rank0 = ortho._tensor_from_bytes([], "float32", b"\x00\x00\x20\x40")
    assert tuple(rank0.shape) == ()
    assert rank0.item() == 2.5


@pytest.mark.parametrize("payload", [b"\x00" * 15, b"\x00" * 17, b""])
def test_tensor_from_bytes_rejects_wrong_length(payload):
    with pytest.raises((RuntimeError, ValueError), match="bytes"):
        ortho._tensor_from_bytes([4], "float32", payload)


def test_tensor_from_bytes_rejects_negative_dimension():
    with pytest.raises((ValueError, RuntimeError)):
        ortho._tensor_from_bytes([-1], "float32", b"")


def test_tensor_from_bytes_guards_shape_overflow():
    """An absurd shape must not wrap around into a small allocation."""
    with pytest.raises((ValueError, RuntimeError), match="overflow"):
        ortho._tensor_from_bytes([2 ** 40, 2 ** 40], "float64", b"")


def test_tensor_to_bytes_requires_cpu_contiguous():
    with pytest.raises(RuntimeError, match="contiguous"):
        ortho._tensor_to_bytes(ortho.zeros([2, 3]).transpose(0, 1))


@requires_cuda
def test_tensor_to_bytes_rejects_cuda():
    with pytest.raises(RuntimeError, match="CPU"):
        ortho._tensor_to_bytes(ortho.zeros([2]).to("cuda"))


def test_contiguous_slice_serializes_only_its_own_payload():
    """data() folds in storage_offset, so a view contributes exactly its own
    logical values - not the rest of the backing buffer."""
    base = _from_numpy(np.arange(10, dtype=np.float32))
    row = base.slice(0, 4, 7)
    assert row.is_contiguous
    assert ortho._tensor_to_bytes(row) == np.arange(4, 7, dtype=np.float32).tobytes()


# archive layout and manifest
def test_archive_is_a_zip_with_json_manifest_and_binary_payloads(tmp_path):
    path = tmp_path / "state.anything"        # extension is irrelevant
    ortho.save_state_dict(_model(), path)

    with open(path, "rb") as handle:
        assert handle.read(4) == b"PK\x03\x04"

    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        assert names[0] == "manifest.json"
        assert all(re.match(r"^tensors/[0-9]{6}\.bin$", n) for n in names[1:])
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        # Payloads are opaque binary, not text or JSON.
        blob = archive.read(names[1])
        assert isinstance(blob, bytes)
        with pytest.raises(ValueError):
            json.loads(blob)

    assert manifest["format"] == FORMAT_ID
    assert manifest["format_version"] == FORMAT_VERSION
    assert set(manifest["sections"]) == {"model", "optimizer", "metadata"}


def test_format_identifier_is_brand_neutral():
    """The checkpoint identifier is intentionally independent of package
    branding, so a rename, vendoring or fork never forces a file migration."""
    assert FORMAT_ID == "ml-runtime-checkpoint"
    assert "orthon" not in FORMAT_ID
    assert FORMAT_VERSION == 1


def test_tensor_record_carries_everything_needed_to_reconstruct(tmp_path):
    path = tmp_path / "rec.bin"
    tensor = _from_numpy(np.arange(6, dtype=np.float64).reshape(2, 3))
    ortho.save_checkpoint(path, model=Holder(tensor))
    record = _manifest(path)["sections"]["model"]["v"][0][1]
    assert record["k"] == "tensor"
    assert record["shape"] == [2, 3]
    assert record["dtype"] == "float64"
    assert record["nbytes"] == 48
    assert record["payload"] == "tensors/000000.bin"
    with zipfile.ZipFile(path) as archive:
        blob = archive.read(record["payload"])
    assert record["sha256"] == hashlib.sha256(blob).hexdigest()
    assert blob == ortho._tensor_to_bytes(tensor)


def test_manifest_does_not_record_device(tmp_path):
    """Device placement is a LOAD policy, not intrinsic tensor-file state."""
    path = tmp_path / "nodev.bin"
    ortho.save_state_dict(_model(), path)
    text = json.dumps(_manifest(path))
    assert "device" not in text
    assert "cuda" not in text
    assert "cpu" not in text


def test_manifest_names_and_dtypes_are_readable_without_executing_code(tmp_path):
    """The format is inspectable with stdlib zipfile + json alone."""
    path = tmp_path / "inspect.bin"
    ortho.save_state_dict(_model(), path)
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    entries = manifest["sections"]["model"]["v"]
    names = [name for name, _ in entries]
    assert names == sorted(names)
    assert names == ["0.bias", "0.weight", "2.bias", "2.weight"]
    for _, record in entries:
        assert record["dtype"] == "float32"
        assert isinstance(record["shape"], list)


def test_state_only_archive_has_no_optimizer_section(tmp_path):
    path = tmp_path / "state.bin"
    ortho.save_state_dict(_model(), path)
    sections = _manifest(path)["sections"]
    assert sections["model"] is not None
    assert sections["optimizer"] is None
    assert sections["metadata"] == {"k": "dict", "v": []}


# deterministic archives
def _sha(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def test_saving_the_same_state_twice_is_byte_identical(tmp_path):
    model = _model()
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    ortho.save_state_dict(model, a)
    ortho.save_state_dict(model, b)
    assert _sha(a) == _sha(b)


def test_full_checkpoint_is_deterministic(tmp_path):
    from orthon.optim import Adam
    model = _model()
    optimizer = Adam(model.parameters(), lr=0.01)
    x = ortho.full([4, 3], 0.25)
    ortho.sum(model(x) * model(x)).backward()
    optimizer.step()
    meta = {"epoch": 1, "tag": "t", "nested": {"a": [1, 2, (3, 4)]}}
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    ortho.save_checkpoint(a, model=model, optimizer=optimizer, metadata=meta)
    ortho.save_checkpoint(b, model=model, optimizer=optimizer, metadata=meta)
    assert _sha(a) == _sha(b)


def test_archive_contains_no_ambient_environment(tmp_path):
    path = tmp_path / "clean.bin"
    ortho.save_state_dict(_model(), path)
    with open(path, "rb") as handle:
        blob = handle.read()
    for leak in (os.getcwd().encode(), str(tmp_path).encode(),
                 b"clean.bin", b"C:\\", b"/home/"):
        assert leak not in blob
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert info.create_system == 0


def test_state_names_are_sorted_in_the_archive(tmp_path):
    """The FILE normalises order; in-memory state_dict is untouched."""
    model = _model()
    path = tmp_path / "sorted.bin"
    ortho.save_state_dict(model, path)
    names = [name for name, _ in _manifest(path)["sections"]["model"]["v"]]
    assert names == sorted(names)
    assert set(model.state_dict()) == set(names)


# exact bit round-trips through the file
@pytest.mark.parametrize("dtype,array", DTYPE_CASES)
def test_file_round_trip_is_bit_exact(dtype, array, tmp_path):
    tensor = _from_numpy(array)
    before = ortho._tensor_to_bytes(tensor)
    path = tmp_path / ("rt_%s.bin" % dtype)
    ortho.save_checkpoint(path, model=Holder(tensor))
    destination = Holder(ortho._tensor_from_bytes(list(tensor.shape), dtype,
                                               b"\x00" * len(before)))
    ortho.load_checkpoint(path, model=destination)
    assert ortho._tensor_to_bytes(destination.tensor) == before


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_low_precision_file_round_trip_is_bit_exact(dtype, tmp_path):
    """Restore copies bytes, it does not re-convert through float32."""
    source = _from_numpy(np.array([1e-5, 65504.0, 1 / 3, -0.0, np.inf],
                                  dtype=np.float32))
    tensor = ortho.cast(source, dtype)
    before = ortho._tensor_to_bytes(tensor)
    path = tmp_path / ("lp_%s.bin" % dtype)
    ortho.save_checkpoint(path, model=Holder(tensor))
    destination = Holder(ortho.cast(ortho.zeros([5]), dtype))
    ortho.load_checkpoint(path, model=destination)
    assert ortho._tensor_to_bytes(destination.tensor) == before


def test_signed_zero_and_non_finite_values_survive(tmp_path):
    array = np.array([0.0, -0.0, np.inf, -np.inf, np.nan], dtype=np.float32)
    tensor = _from_numpy(array)
    path = tmp_path / "nonfinite.bin"
    ortho.save_checkpoint(path, model=Holder(tensor))
    destination = Holder(ortho.zeros([5]))
    ortho.load_checkpoint(path, model=destination)
    got = ortho._tensor_to_bytes(destination.tensor)
    assert got == array.tobytes()
    words = np.frombuffer(got, dtype=np.uint32)
    assert words[0] == 0x00000000            # +0.0
    assert words[1] == 0x80000000            # -0.0, sign bit intact
    assert words[2] == 0x7F800000            # +inf
    assert words[3] == 0xFF800000            # -inf


def test_nan_payload_survives(tmp_path):
    """A signalling NaN keeps its payload; a Tensor -> float -> NumPy
    pipeline would quiet 0x7F800001 to 0x7FC00001."""
    source = np.array([0x7F800001, 0x7FC0DEAD], dtype=np.uint32).view(np.float32)
    tensor = _from_numpy(source)
    path = tmp_path / "nan.bin"
    ortho.save_checkpoint(path, model=Holder(tensor))
    destination = Holder(ortho.zeros([2]))
    ortho.load_checkpoint(path, model=destination)
    words = np.frombuffer(ortho._tensor_to_bytes(destination.tensor), dtype=np.uint32)
    assert list(words) == [0x7F800001, 0x7FC0DEAD]


def test_non_contiguous_source_is_stored_in_logical_order(tmp_path):
    base = _from_numpy(np.arange(6, dtype=np.float32).reshape(2, 3))
    view = base.transpose(0, 1)
    assert not view.is_contiguous
    path = tmp_path / "view.bin"
    ortho.save_checkpoint(path, model=Holder(view))
    record = _manifest(path)["sections"]["model"]["v"][0][1]
    assert record["shape"] == [3, 2]
    assert record["nbytes"] == 24
    destination = Holder(ortho.zeros([3, 2]))
    ortho.load_checkpoint(path, model=destination)
    assert destination.tensor.tolist() == [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]


@requires_cuda
def test_cuda_tensor_serialization(tmp_path):
    array = np.array([1.0, -0.0, np.inf, 2.5], dtype=np.float32)
    tensor = _from_numpy(array).to("cuda")
    path = tmp_path / "cuda.bin"
    ortho.save_checkpoint(path, model=Holder(tensor))
    with zipfile.ZipFile(path) as archive:
        assert archive.read("tensors/000000.bin") == array.tobytes()
    # restores onto CPU or CUDA, driven by the destination
    on_cpu = Holder(ortho.zeros([4]))
    ortho.load_checkpoint(path, model=on_cpu)
    assert ortho._tensor_to_bytes(on_cpu.tensor) == array.tobytes()
    on_gpu = Holder(ortho.zeros([4]).to("cuda"))
    ortho.load_checkpoint(path, model=on_gpu)
    assert on_gpu.tensor.device.is_cuda()
    assert _raw(on_gpu.tensor) == array.tobytes()


# saving is observational
def test_saving_does_not_mutate_the_source(tmp_path):
    tensor = _from_numpy(np.arange(4, dtype=np.float32))
    tensor.requires_grad_(True)
    ortho.sum(tensor * 2.0).backward()

    before = {
        "version": tensor._storage_version,
        "requires_grad": tensor.requires_grad,
        "has_grad_fn": tensor._has_grad_fn,
        "is_leaf": tensor.is_leaf,
        "device": tensor.device,
        "dtype": tensor.dtype,
        "impl": tensor._impl_id,
        "grad": tensor.grad.tolist(),
        "bytes": ortho._tensor_to_bytes(tensor),
    }
    ortho.save_checkpoint(tmp_path / "obs.bin", model=Holder(tensor))
    assert tensor._storage_version == before["version"]     # untouched
    assert tensor.requires_grad == before["requires_grad"]
    assert tensor._has_grad_fn == before["has_grad_fn"]
    assert tensor.is_leaf == before["is_leaf"]
    assert tensor.device == before["device"]
    assert tensor.dtype == before["dtype"]
    assert tensor._impl_id == before["impl"]
    assert tensor.grad.tolist() == before["grad"]
    assert ortho._tensor_to_bytes(tensor) == before["bytes"]


def test_file_is_a_snapshot_not_a_live_reference(tmp_path):
    model = nn.Linear(2, 2)
    model.weight.fill_(1.0)
    path = tmp_path / "snap.bin"
    ortho.save_state_dict(model, path)

    model.weight.fill_(9.0)                      # mutate AFTER saving
    restored = nn.Linear(2, 2)
    ortho.load_state_dict(restored, path)
    assert restored.weight.tolist() == [1.0] * 4


def test_state_dict_returns_live_references(tmp_path):
    """state_dict() hands back the live tensors, not copies."""
    model = nn.Linear(2, 2)
    state = model.state_dict()
    model.weight.fill_(7.0)
    assert state["weight"].tolist() == [7.0] * 4      # the same tensor
    assert state["weight"]._impl_id == model.weight._impl_id
    doc = nn.Module.state_dict.__doc__.lower()
    assert "live" in doc
    assert "snapshot" in doc


# the safe recursive value codec
SAFE_VALUES = [
    None, True, False, 0, -17, 2 ** 62, 0.0, -0.5, 1e300,
    "", "text", "unicode \u00e9\u4e2d",
    [], [1, "a", None], (), (1, 2), [(1, 2), [3, 4]],
    {}, {"a": 1}, {"a": {"b": [1, (2, 3)]}},
]


@pytest.mark.parametrize("value", SAFE_VALUES)
def test_metadata_round_trip(value, tmp_path):
    path = tmp_path / "meta.bin"
    ortho.save_checkpoint(path, metadata={"v": value})
    assert ortho.load_checkpoint(path) == {"v": value}


def test_tuple_and_list_stay_distinct(tmp_path):
    path = tmp_path / "seq.bin"
    ortho.save_checkpoint(path, metadata={"t": (1, 2), "l": [1, 2],
                                       "mixed": [(1,), [2]]})
    loaded = ortho.load_checkpoint(path)
    assert isinstance(loaded["t"], tuple)
    assert isinstance(loaded["l"], list)
    assert isinstance(loaded["mixed"][0], tuple)
    assert isinstance(loaded["mixed"][1], list)


def test_non_finite_python_metadata(tmp_path):
    """Tagged, so the manifest never needs JSON's NaN/Infinity."""
    path = tmp_path / "nonfinite.bin"
    ortho.save_checkpoint(path, metadata={"p": float("inf"), "n": float("-inf"),
                                       "q": float("nan")})
    raw = json.dumps(_manifest(path))
    assert "NaN" not in raw and "Infinity" not in raw
    loaded = ortho.load_checkpoint(path)
    assert loaded["p"] == float("inf")
    assert loaded["n"] == float("-inf")
    assert loaded["q"] != loaded["q"]


def test_bool_is_not_confused_with_int(tmp_path):
    path = tmp_path / "bool.bin"
    ortho.save_checkpoint(path, metadata={"b": True, "i": 1})
    loaded = ortho.load_checkpoint(path)
    assert loaded["b"] is True
    assert isinstance(loaded["i"], int) and not isinstance(loaded["i"], bool)


def test_reserved_key_collision_is_impossible(tmp_path):
    """Dicts encode as item LISTS, so user keys can never look like tags."""
    hostile = {"k": "tensor", "v": "x", "shape": [1], "payload": "../../evil"}
    path = tmp_path / "hostile.bin"
    ortho.save_checkpoint(path, metadata=hostile)
    assert ortho.load_checkpoint(path) == hostile


def test_tensors_nested_in_metadata_round_trip(tmp_path):
    tensor = _from_numpy(np.array([1.5, -2.5], dtype=np.float32))
    path = tmp_path / "nested.bin"
    ortho.save_checkpoint(path, metadata={"buffers": [tensor, None]})
    loaded = ortho.load_checkpoint(path)
    assert ortho._tensor_to_bytes(loaded["buffers"][0]) == ortho._tensor_to_bytes(tensor)
    assert loaded["buffers"][1] is None


@pytest.mark.parametrize("value", [object(), set(), b"bytes", complex(1, 2),
                                   lambda: None, type, Ellipsis])
def test_unsupported_value_types_raise_type_error(value, tmp_path):
    with pytest.raises(TypeError, match="unsupported value type"):
        ortho.save_checkpoint(tmp_path / "bad.bin", metadata={"x": value})


def test_unsupported_value_error_names_the_path(tmp_path):
    with pytest.raises(TypeError) as info:
        ortho.save_checkpoint(tmp_path / "bad.bin",
                           metadata={"outer": {"inner": [1, object()]}})
    assert "metadata.outer.inner[1]" in str(info.value)


def test_non_string_dict_keys_rejected(tmp_path):
    with pytest.raises(TypeError, match="dict key"):
        ortho.save_checkpoint(tmp_path / "bad.bin", metadata={1: "one"})


# pickle is gone, and legacy files are refused without unpickling
def _strip_py_comments_and_docstrings(text):
    """Remove '#' comments and triple-quoted strings.

    checkpoint.py documents at length what it REPLACED, quoting the old code,
    so a raw substring search would find the very constructs it must prove
    absent.
    """
    text = re.sub(r'"""(?:.|\n)*?"""', "", text)
    text = re.sub(r"'''(?:.|\n)*?'''", "", text)
    return re.sub(r"#[^\n]*", "", text)


def _runtime_python_sources():
    for dirpath, dirnames, names in os.walk(os.path.join(REPO, "python")):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in sorted(names):
            if name.endswith(".py"):
                full = os.path.join(dirpath, name)
                yield os.path.relpath(full, REPO).replace("\\", "/"), full


# `eval(` and `exec(` are matched with leading-boundary guards so the Module
# training-mode API - `model.eval()` and `def eval(self)` - is not mistaken for
# the builtins.
_UNSAFE_PATTERNS = [
    ("pickle import", re.compile(r"^\s*(import\s+pickle|from\s+pickle\s)", re.M)),
    ("pickle.load", re.compile(r"\bpickle\.loads?\s*\(")),
    ("pickle.dump", re.compile(r"\bpickle\.dumps?\s*\(")),
    ("marshal", re.compile(r"\bmarshal\.")),
    ("builtin eval", re.compile(r"(?<![\w.])(?<!def )eval\s*\(")),
    ("builtin exec", re.compile(r"(?<![\w.])(?<!def )exec\s*\(")),
]


def test_runtime_contains_no_pickle():
    """The runtime has no pickle code path at all - not an unused one."""
    offenders = []
    for rel, full in _runtime_python_sources():
        with open(full, encoding="utf-8") as handle:
            body = _strip_py_comments_and_docstrings(handle.read())
        for label, pattern in _UNSAFE_PATTERNS:
            if pattern.search(body):
                offenders.append("%s: %s" % (rel, label))
    assert offenders == []


def test_exactly_one_serialization_module_exists():
    assert not os.path.exists(
        os.path.join(REPO, "python", "orthon", "serialization.py"))
    with pytest.raises(ImportError):
        import orthon.serialization  # noqa: F401


def test_all_four_file_apis_come_from_one_module():
    for fn in (ortho.save_state_dict, ortho.load_state_dict,
               ortho.save_checkpoint, ortho.load_checkpoint):
        assert fn.__module__ == "orthon.checkpoint"


def test_checkpoint_module_does_not_import_numpy():
    """Stdlib only; reconstructing a tensor must not need NumPy."""
    import orthon.checkpoint as module
    with open(module.__file__, encoding="utf-8") as handle:
        body = handle.read()
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            assert "numpy" not in stripped, stripped
    imported = {line.strip().split()[1].split(".")[0]
                for line in body.splitlines()
                if line.strip().startswith("import ")}
    assert imported <= {"hashlib", "json", "os", "re", "tempfile",
                        "zipfile", "orthon"}, imported


def test_checkpoint_serialization_has_no_tolist():
    """Hard gate. Encoding must never go through Python floats."""
    import orthon.checkpoint as module
    with open(module.__file__, encoding="utf-8") as handle:
        code = _strip_py_comments_and_docstrings(handle.read())
    from pathlib import Path
    for source in (Path(module.__file__).parent / "_serialization").glob("*.py"):
        code += _strip_py_comments_and_docstrings(source.read_text())
    assert ".tolist()" not in code
    assert "set_item" not in code
    assert "_tensor_to_bytes" in code
    assert "_tensor_from_bytes" in code


def test_legacy_pickle_is_rejected_without_unpickling(tmp_path, monkeypatch):
    path = tmp_path / "legacy.pkl"
    with open(path, "wb") as handle:
        pickle.dump({"version": 1, "model": None}, handle)

    def explode(*args, **kwargs):
        raise AssertionError("pickle.load must never be called")

    monkeypatch.setattr(pickle, "load", explode)
    monkeypatch.setattr(pickle, "loads", explode)
    with pytest.raises(RuntimeError, match="legacy pickle"):
        ortho.load_checkpoint(path)


@pytest.mark.parametrize("protocol", [0, 1, 2, 3, 4, 5])
def test_pickles_of_every_protocol_are_rejected(protocol, tmp_path):
    path = tmp_path / ("p%d.pkl" % protocol)
    with open(path, "wb") as handle:
        pickle.dump({"a": 1}, handle, protocol=protocol)
    with pytest.raises(RuntimeError, match="checkpoint:"):
        ortho.load_checkpoint(path)


def test_detection_ignores_the_file_extension(tmp_path):
    """A valid archive named .pkl loads; a pickle named .bin does not."""
    good = tmp_path / "looks_like_a_pickle.pkl"
    ortho.save_state_dict(_model(), good)
    ortho.load_state_dict(_model(), good)

    bad = tmp_path / "looks_like_an_archive.bin"
    with open(bad, "wb") as handle:
        pickle.dump({"a": 1}, handle)
    with pytest.raises(RuntimeError):
        ortho.load_checkpoint(bad)


# corruption detection
@pytest.fixture()
def sample(tmp_path):
    path = tmp_path / "sample.bin"
    ortho.save_checkpoint(path, model=_model(), metadata={"epoch": 1})
    return path


def test_random_file_rejected(tmp_path):
    path = tmp_path / "random.bin"
    with open(path, "wb") as handle:
        handle.write(os.urandom(512))
    with pytest.raises(RuntimeError, match="checkpoint:"):
        ortho.load_checkpoint(path)


def test_truncated_zip_rejected(tmp_path, sample):
    path = tmp_path / "truncated.bin"
    with open(sample, "rb") as handle:
        blob = handle.read()
    with open(path, "wb") as handle:
        handle.write(blob[:len(blob) // 2])
    with pytest.raises(RuntimeError, match="checkpoint:"):
        ortho.load_checkpoint(path)


def test_missing_manifest_rejected(tmp_path, sample):
    path = _rewrite(sample, tmp_path / "nomanifest.bin",
                    mutate_manifest=lambda m: None)
    with pytest.raises(RuntimeError, match="no manifest"):
        ortho.load_checkpoint(path)


def test_wrong_format_id_rejected(tmp_path, sample):
    def mutate(manifest):
        manifest["format"] = "some-other-format"
        return manifest
    path = _rewrite(sample, tmp_path / "wrongid.bin", mutate_manifest=mutate)
    with pytest.raises(RuntimeError, match="unsupported format"):
        ortho.load_checkpoint(path)


def test_unsupported_version_rejected(tmp_path, sample):
    def mutate(manifest):
        manifest["format_version"] = FORMAT_VERSION + 1
        return manifest
    path = _rewrite(sample, tmp_path / "newer.bin", mutate_manifest=mutate)
    with pytest.raises(RuntimeError, match="unsupported version"):
        ortho.load_checkpoint(path)


def test_missing_sections_rejected(tmp_path, sample):
    def mutate(manifest):
        del manifest["sections"]["metadata"]
        return manifest
    path = _rewrite(sample, tmp_path / "nosection.bin", mutate_manifest=mutate)
    with pytest.raises(RuntimeError, match="section"):
        ortho.load_checkpoint(path)


def test_missing_tensor_payload_rejected(tmp_path, sample):
    def drop(blobs, names):
        victim = names[1]
        del blobs[victim]
        return blobs, [n for n in names if n != victim]
    path = _rewrite(sample, tmp_path / "nopayload.bin", mutate_entries=drop)
    with pytest.raises(RuntimeError, match="missing tensor payload"):
        ortho.load_checkpoint(path, model=_model())


def test_payload_byte_count_mismatch_rejected(tmp_path, sample):
    def shrink(blobs, names):
        blobs[names[1]] = blobs[names[1]][:-4]
        return blobs, names
    path = _rewrite(sample, tmp_path / "short.bin", mutate_entries=shrink)
    with pytest.raises(RuntimeError, match="manifest declares"):
        ortho.load_checkpoint(path, model=_model())


def test_payload_checksum_mismatch_rejected(tmp_path, sample):
    def flip(blobs, names):
        blob = bytearray(blobs[names[1]])
        blob[0] ^= 0xFF
        blobs[names[1]] = bytes(blob)
        return blobs, names
    path = _rewrite(sample, tmp_path / "flipped.bin", mutate_entries=flip)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        ortho.load_checkpoint(path, model=_model())


def test_duplicate_payload_reference_rejected(tmp_path, sample):
    def mutate(manifest):
        entries = manifest["sections"]["model"]["v"]
        entries[1][1]["payload"] = entries[0][1]["payload"]
        entries[1][1]["sha256"] = entries[0][1]["sha256"]
        entries[1][1]["nbytes"] = entries[0][1]["nbytes"]
        entries[1][1]["shape"] = entries[0][1]["shape"]
        return manifest
    path = _rewrite(sample, tmp_path / "dup.bin", mutate_manifest=mutate)
    with pytest.raises(RuntimeError, match="more than once"):
        ortho.load_checkpoint(path, model=_model())


def test_malformed_dtype_rejected(tmp_path, sample):
    def mutate(manifest):
        manifest["sections"]["model"]["v"][0][1]["dtype"] = "float999"
        return manifest
    path = _rewrite(sample, tmp_path / "dtype.bin", mutate_manifest=mutate)
    with pytest.raises(RuntimeError, match="cannot construct tensor"):
        ortho.load_checkpoint(path, model=_model())


@pytest.mark.parametrize("shape", [[-1], "notalist", [1.5], [True]])
def test_malformed_shape_rejected(shape, tmp_path, sample):
    def mutate(manifest):
        manifest["sections"]["model"]["v"][0][1]["shape"] = shape
        return manifest
    path = _rewrite(sample, tmp_path / "shape.bin", mutate_manifest=mutate)
    with pytest.raises(RuntimeError, match="shape"):
        ortho.load_checkpoint(path, model=_model())


def test_shape_inconsistent_with_payload_rejected(tmp_path, sample):
    """nbytes agrees with the blob, but not with shape * dtype_size."""
    def mutate(manifest):
        record = manifest["sections"]["model"]["v"][0][1]
        record["shape"] = [record["nbytes"] // 4 + 1]
        return manifest
    path = _rewrite(sample, tmp_path / "inconsistent.bin", mutate_manifest=mutate)
    with pytest.raises(RuntimeError, match="cannot construct tensor"):
        ortho.load_checkpoint(path, model=_model())


@pytest.mark.parametrize("kind", ["bogus", None, 42, "eval"])
def test_unsupported_encoded_kind_rejected(kind, tmp_path, sample):
    def mutate(manifest):
        manifest["sections"]["metadata"] = {"k": kind, "v": 1}
        return manifest
    path = _rewrite(sample, tmp_path / "kind.bin", mutate_manifest=mutate)
    with pytest.raises(RuntimeError, match="unsupported encoded value kind"):
        ortho.load_checkpoint(path)


@pytest.mark.parametrize("name", [
    "../../evil", "/absolute/evil", "..\\evil", "tensors/../evil",
    "tensors/0.bin", "tensors/0000000.bin", "evil.bin", "tensors/000000.txt",
])
def test_illegal_payload_names_rejected(name, tmp_path, sample):
    """An allowlist, so traversal never reaches a filesystem call."""
    def mutate(manifest):
        manifest["sections"]["model"]["v"][0][1]["payload"] = name
        return manifest
    path = _rewrite(sample, tmp_path / "traversal.bin", mutate_manifest=mutate)
    with pytest.raises(RuntimeError, match="illegal tensor payload name"):
        ortho.load_checkpoint(path, model=_model())


def test_loader_never_extracts_to_disk(monkeypatch, sample):
    def explode(*args, **kwargs):
        raise AssertionError("the loader must never extract to disk")
    monkeypatch.setattr(zipfile.ZipFile, "extractall", explode)
    monkeypatch.setattr(zipfile.ZipFile, "extract", explode)
    ortho.load_checkpoint(sample, model=_model())


def test_absurd_declared_size_is_rejected_before_allocating(tmp_path, sample):
    def mutate(manifest):
        record = manifest["sections"]["model"]["v"][0][1]
        record["shape"] = [2 ** 40]
        record["nbytes"] = 2 ** 42
        return manifest
    path = _rewrite(sample, tmp_path / "absurd.bin", mutate_manifest=mutate)
    with pytest.raises(RuntimeError, match="manifest declares"):
        ortho.load_checkpoint(path, model=_model())


# validate before mutating
def test_late_corruption_leaves_the_model_untouched(tmp_path):
    """The LAST tensor is corrupted; the FIRST must not have been written."""
    good = tmp_path / "good.bin"
    source = _model()
    for param in source.parameters():
        param.fill_(3.0)
    ortho.save_checkpoint(good, model=source)

    def flip_last(blobs, names):
        blob = bytearray(blobs[names[-1]])
        blob[0] ^= 0xFF
        blobs[names[-1]] = bytes(blob)
        return blobs, names
    path = _rewrite(good, tmp_path / "late.bin", mutate_entries=flip_last)

    destination = _model()
    for param in destination.parameters():
        param.fill_(-1.0)
    before = {k: v.tolist() for k, v in destination.state_dict().items()}

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        ortho.load_checkpoint(path, model=destination)

    after = {k: v.tolist() for k, v in destination.state_dict().items()}
    assert after == before
    assert all(value == -1.0 for values in after.values() for value in values)


def test_shape_mismatch_leaves_the_model_untouched(tmp_path):
    path = tmp_path / "wrongshape.bin"
    ortho.save_state_dict(nn.Linear(3, 4), path)
    destination = nn.Linear(3, 8)
    destination.weight.fill_(-2.0)
    before = destination.weight.tolist()
    with pytest.raises((ValueError, KeyError, RuntimeError)):
        ortho.load_state_dict(destination, path)
    assert destination.weight.tolist() == before


# atomic save
def test_failed_save_leaves_the_previous_file_intact(tmp_path):
    path = tmp_path / "keep.bin"
    ortho.save_state_dict(_model(), path)
    with open(path, "rb") as handle:
        before = handle.read()

    with pytest.raises(TypeError):
        ortho.save_checkpoint(path, model=_model(), metadata={"bad": object()})

    with open(path, "rb") as handle:
        assert handle.read() == before
    assert [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"] == []


def test_failed_save_creates_no_file_at_a_fresh_path(tmp_path):
    path = tmp_path / "never.bin"
    with pytest.raises(TypeError):
        ortho.save_checkpoint(path, metadata={"bad": object()})
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_save_into_a_missing_directory_is_a_clear_error(tmp_path):
    with pytest.raises(RuntimeError, match="destination directory"):
        ortho.save_state_dict(_model(), tmp_path / "nope" / "x.bin")


# path types
def test_str_and_pathlike_are_both_accepted(tmp_path):
    model = _model()
    as_path = tmp_path / "p.bin"
    ortho.save_state_dict(model, as_path)              # pathlib.Path
    ortho.load_state_dict(_model(), as_path)
    as_str = str(tmp_path / "s.bin")
    ortho.save_state_dict(model, as_str)               # str
    ortho.load_state_dict(_model(), as_str)


@pytest.mark.parametrize("bad", [123, None, b"bytes-path", io.BytesIO()])
def test_bad_path_types_raise_type_error(bad, tmp_path):
    with pytest.raises(TypeError, match="str or os.PathLike"):
        ortho.save_state_dict(_model(), bad)


# structural size proof
def test_payload_is_raw_and_the_manifest_stays_small(tmp_path):
    n = 20000
    tensor = _from_numpy(np.arange(n, dtype=np.float32) * 0.5)
    path = tmp_path / "big.bin"
    ortho.save_checkpoint(path, model=Holder(tensor))

    with zipfile.ZipFile(path) as archive:
        payload = archive.getinfo("tensors/000000.bin")
        manifest_info = archive.getinfo("manifest.json")
        blob = archive.read("tensors/000000.bin")

    assert payload.file_size == n * 4               # exactly numel * dtype_size
    assert blob == ortho._tensor_to_bytes(tensor)
    assert manifest_info.file_size < 512            # metadata, not data
    # The old pickle stored every element as a Python float object and ran
    # ~2.25x the raw size; the archive is a small constant over it.
    assert os.path.getsize(path) < n * 4 * 1.05

    manifest_text = json.dumps(_manifest(path))
    assert "0.5" not in manifest_text               # no element values in JSON
    assert len(manifest_text) < 512


# compressed payload rejection
def test_compressed_tensor_payload_rejected(tmp_path, sample):
    """v1 tensor payloads are ZIP_STORED; a DEFLATED payload is refused
    before any read, closing the decompression-amplification hole (a tiny
    archive may not expand into a huge allocation)."""
    path = tmp_path / "deflated.bin"
    with zipfile.ZipFile(sample) as archive:
        names = archive.namelist()
        blobs = {name: archive.read(name) for name in names}
    victim = next(n for n in names if n.startswith("tensors/"))
    with zipfile.ZipFile(path, "w") as archive:
        for name in names:
            compress = (zipfile.ZIP_DEFLATED if name == victim
                        else zipfile.ZIP_STORED)
            archive.writestr(name, blobs[name], compress_type=compress)
    with pytest.raises(RuntimeError, match="stored uncompressed"):
        ortho.load_checkpoint(path, model=_model())
