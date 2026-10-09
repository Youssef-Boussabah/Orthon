import pytest

import orthon as ortho


@pytest.mark.parametrize("spec,kind,index", [
    ("cpu", ortho.DeviceType.CPU, 0), ("cuda", ortho.DeviceType.CUDA, 0),
    ("cuda:0", ortho.DeviceType.CUDA, 0), ("cuda:7", ortho.DeviceType.CUDA, 7),
])
def test_device_values_can_be_described_without_allocating_on_hardware(spec, kind, index):
    parsed = ortho.Device(spec)
    explicit = ortho.Device(kind, index)
    assert parsed == explicit
    assert hash(parsed) == hash(explicit)
    assert parsed.type == kind
    assert parsed.index == index
    assert {parsed: "value"}[explicit] == "value"


@pytest.mark.parametrize("spec", ["cpu:0", "CUDA", "cuda:-1", "cuda:", "cuda:1x", "gpu", ""])
def test_invalid_device_strings_are_rejected(spec, typed_tensor):
    with pytest.raises(ValueError):
        ortho.Device(spec)
    with pytest.raises(ValueError):
        typed_tensor([1]).to(spec)


@pytest.mark.parametrize("kind,index", [(ortho.DeviceType.CPU, 1), (ortho.DeviceType.CUDA, -1)])
def test_device_ordinals_are_validated_without_runtime_execution(kind, index):
    with pytest.raises(ValueError):
        ortho.Device(kind, index)


def test_cpu_copies_preserve_every_storage_dtype(storage_dtype, typed_tensor, assert_output):
    source = typed_tensor([[0, 1], [1, 0]], dtype=storage_dtype)
    assert_output(source.to("cpu"), [[0, 1], [1, 0]], dtype=storage_dtype, device="cpu")
    assert_output(ortho.to(source, ortho.Device("cpu")), [[0, 1], [1, 0]],
                  dtype=storage_dtype, device="cpu")


@pytest.mark.parametrize("layout", ["contiguous", "transpose"])
def test_cuda_round_trip_preserves_every_storage_dtype(
    cuda_device, storage_dtype, typed_tensor, assert_output, layout,
):
    values = [[0, 1, 1], [1, 0, 0]]
    source = typed_tensor(values, dtype=storage_dtype)
    if layout == "transpose":
        source, values = source.transpose(0, 1), [[0, 1], [1, 0], [1, 0]]
    moved = source.to(cuda_device)
    assert_output(moved, values, dtype=storage_dtype, device=cuda_device)
    assert_output(moved.to("cpu"), values, dtype=storage_dtype, device="cpu")


@pytest.mark.parametrize("name", ["add", "sub", "mul", "div", "matmul"])
@pytest.mark.parametrize("reverse", [False, True])
def test_tensor_operations_reject_cross_device_operands(cuda_device, typed_tensor, name, reverse):
    cpu = typed_tensor([[1, 2], [3, 4]])
    cuda = cpu.to(cuda_device)
    with pytest.raises(RuntimeError):
        getattr(ortho, name)(cuda, cpu) if reverse else getattr(ortho, name)(cpu, cuda)


def test_cpu_only_build_rejects_cuda_movement_instead_of_falling_back(typed_tensor):
    if ortho.cuda_enabled():
        pytest.skip("contract applies to CPU-only builds")
    source = typed_tensor([1, 2])
    with pytest.raises(RuntimeError):
        source.to("cuda:0")
    assert source.device == ortho.Device("cpu")
