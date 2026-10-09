import numpy as np
import pytest

import orthon as ortho
from orthon.nn import Module, Parameter

from .workload import RegressionModel, value_tree


@pytest.mark.parametrize("shape", [(), (0,), (2, 2), (3, 0, 2)])
def test_all_storage_dtypes_and_shapes_survive_load_and_resave_without_reencoding(
    typed_tensor, storage_dtype, operator_device, checkpoint_archive, tmp_path, shape,
):
    size = int(np.prod(shape))
    values = np.arange(size).reshape(shape)
    if storage_dtype.startswith("float") or storage_dtype == "bfloat16":
        values = values / 3. - 0.5
    source = typed_tensor(values, dtype=storage_dtype, device=operator_device)
    path, again = tmp_path / "dtype.ckpt", tmp_path / "resaved.ckpt"
    ortho.save_checkpoint(path, metadata={"tensor": source})
    restored = ortho.load_checkpoint(path)["tensor"]
    assert restored.shape == shape
    assert restored.dtype == storage_dtype
    assert restored.device.is_cpu()
    assert restored.is_leaf and not restored.requires_grad
    ortho.save_checkpoint(again, metadata={"tensor": restored})
    first, second = checkpoint_archive(path), checkpoint_archive(again)
    a, b = first.records("metadata")[0], second.records("metadata")[0]
    assert "device" not in a
    assert a["shape"] == list(shape)
    assert a["dtype"] == storage_dtype
    assert first.payloads[a["payload"]] == second.payloads[b["payload"]]


@pytest.mark.parametrize("dtype,words", [
    ("float32", [0x00000000, 0x80000000, 0x7F800001, 0x7FC12345, 0x7F800000, 0xFF800000]),
    ("float64", [0x0000000000000000, 0x8000000000000000, 0x7FF0000000000001,
                 0x7FF8123456789ABC, 0x7FF0000000000000, 0xFFF0000000000000]),
])
def test_special_float_bits_survive_public_load_and_resave(
    dtype, words, tmp_path, checkpoint_archive,
):
    array = np.array(words, dtype="uint32" if dtype == "float32" else "uint64").view(dtype)
    source = ortho.from_numpy(array)
    path, again = tmp_path / "bits.ckpt", tmp_path / "bits-again.ckpt"
    ortho.save_checkpoint(path, metadata={"tensor": source})
    restored = ortho.load_checkpoint(path)["tensor"]
    ortho.save_checkpoint(again, metadata={"tensor": restored})
    archive = checkpoint_archive(again)
    record = archive.records("metadata")[0]
    assert restored.dtype == dtype
    assert archive.payloads[record["payload"]] == array.tobytes()


@pytest.mark.parametrize("source_dtype,target_dtype", [
    ("float32", "float64"), ("float64", "float32"),
    ("float32", "float16"), ("float32", "bfloat16"),
    ("float16", "float32"), ("bfloat16", "float32"),
])
def test_existing_model_dtype_is_authoritative_when_loading_valid_different_dtype(
    typed_tensor, assert_output, tmp_path, source_dtype, target_dtype,
):
    source, target = Module(), Module()
    source.weight = Parameter(typed_tensor([[1.5, -2.25]], dtype=source_dtype))
    source.register_buffer("offset", typed_tensor([0.125], dtype=source_dtype))
    target.weight = Parameter(typed_tensor([[0., 0.]], dtype=target_dtype))
    target.register_buffer("offset", typed_tensor([0.], dtype=target_dtype))
    held = target.weight
    path = tmp_path / "conversion.ckpt"
    ortho.save_state_dict(source, path)
    ortho.load_state_dict(target, path)
    assert target.weight is held
    assert target.weight.requires_grad and target.weight.is_leaf
    assert_output(target.weight, [[1.5, -2.25]], dtype=target_dtype, device="cpu")
    assert_output(target.offset, [0.125], dtype=target_dtype, device="cpu")


@pytest.mark.parametrize("source_device,target_device", [
    ("cpu", "cpu"), ("cpu", "cuda:0"), ("cuda:0", "cpu"), ("cuda:0", "cuda:0"),
])
def test_default_file_load_preserves_destination_device_and_parameter_identity(
    request, tmp_path, source_device, target_device,
):
    if "cuda:0" in (source_device, target_device):
        request.getfixturevalue("cuda_device")
    source, target = RegressionModel(device=source_device), RegressionModel(device=target_device)
    held = list(target.parameters())
    path = tmp_path / "placement.ckpt"
    ortho.save_state_dict(source, path)
    ortho.load_state_dict(target, path)
    assert all(current is old for current, old in zip(target.parameters(), held))
    for name, value in target.state_dict().items():
        expected = source.state_dict()[name]
        assert value.device == ortho.Device(target_device)
        assert value.dtype == expected.dtype
        assert value.shape == expected.shape
        assert value.tolist() == expected.tolist()


@pytest.mark.parametrize("map_location", ["cpu", ortho.Device("cpu")])
def test_explicit_cpu_placement_restores_model_values(tmp_path, map_location):
    source, target = RegressionModel(), RegressionModel()
    source.weight.mul_(2.)
    path = tmp_path / "explicit.ckpt"
    ortho.save_state_dict(source, path)
    ortho.load_state_dict(target, path, map_location=map_location)
    assert value_tree(target.state_dict()) == value_tree(source.state_dict())
    assert all(p.device.is_cpu() for p in target.parameters())


def test_cpu_only_build_rejects_cuda_target_before_applying_model(tmp_path):
    if ortho.cuda_enabled():
        pytest.skip("CPU-only build rejection is inapplicable to a CUDA-enabled package")
    source, target = RegressionModel(), RegressionModel()
    source.weight.mul_(2.)
    held = list(target.parameters())
    before = value_tree(target.state_dict())
    path = tmp_path / "unavailable.ckpt"
    ortho.save_state_dict(source, path)
    with pytest.raises(RuntimeError):
        ortho.load_state_dict(target, path, map_location="cuda:0")
    assert value_tree(target.state_dict()) == before
    assert all(current is old for current, old in zip(target.parameters(), held))


def test_cuda_enabled_build_without_runtime_rejects_target_before_applying_model(tmp_path):
    if not ortho.cuda_enabled():
        pytest.skip("unavailable CUDA runtime rejection needs a CUDA-enabled package")
    try:
        ortho.zeros([1]).to("cuda:0")
    except RuntimeError as error:
        reasons = ("no cuda-capable device", "cuda driver version is insufficient",
                   "initialization error", "unsupported display driver / cuda driver combination")
        if not any(reason in str(error).lower() for reason in reasons):
            raise
    else:
        pytest.skip("CUDA runtime is available; this rejection case does not apply")
    source, target = RegressionModel(), RegressionModel()
    source.weight.mul_(2.)
    held = list(target.parameters())
    before = value_tree(target.state_dict())
    path = tmp_path / "unavailable-runtime.ckpt"
    ortho.save_state_dict(source, path)
    with pytest.raises(RuntimeError):
        ortho.load_state_dict(target, path, map_location="cuda:0")
    assert value_tree(target.state_dict()) == before
    assert all(current is old for current, old in zip(target.parameters(), held))
