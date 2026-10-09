import pytest


@pytest.mark.parametrize("layout", ["contiguous", "transpose", "slice"])
def test_same_device_to_makes_an_independent_copy_in_logical_order(tensor, assert_tensor, layout):
    base = tensor([[1, 2, 3], [4, 5, 6]])
    source, expected = {
        "contiguous": lambda: (base, [[1, 2, 3], [4, 5, 6]]),
        "transpose": lambda: (base.transpose(0, 1), [[1, 4], [2, 5], [3, 6]]),
        "slice": lambda: (base.slice(1, 1, 3), [[2, 3], [5, 6]]),
    }[layout]()
    copied = source.to(source.device)
    assert_tensor(copied, expected)
    assert copied.is_contiguous is True
    assert copied.dtype == source.dtype
    assert copied.device == source.device

    copied[0, 0].mul_(10)
    assert_tensor(source, expected)
    source[0, 0].mul_(3)
    assert copied[0, 0].item() == expected[0][0] * 10


@pytest.mark.parametrize("operation", ["contiguous", "same_dtype_cast"])
def test_normalizing_a_contiguous_tensor_keeps_its_shared_values(tensor, operation):
    source = tensor([[1, 2], [3, 4]])
    normalized = source.contiguous() if operation == "contiguous" else source.cast(source.dtype)
    normalized[1, 0].mul_(10)
    assert source[1, 0].item() == 30
    source[0, 1].mul_(5)
    assert normalized[0, 1].item() == 10


@pytest.mark.parametrize("operation", ["contiguous", "same_dtype_cast"])
def test_normalizing_a_strided_tensor_makes_an_independent_copy(tensor, assert_tensor, operation):
    base = tensor([[1, 2, 3], [4, 5, 6]])
    source = base.transpose(0, 1)
    normalized = source.contiguous() if operation == "contiguous" else source.cast(source.dtype)
    assert normalized.is_contiguous is True
    normalized.mul_(2)
    assert_tensor(base, [[1, 2, 3], [4, 5, 6]])
    source[1, 0].mul_(10)
    assert_tensor(normalized, [[2, 8], [4, 10], [6, 12]])


def test_a_dtype_conversion_retains_its_own_snapshot(tensor, assert_tensor):
    source = tensor([[2, 3], [4, 5]])
    converted = source.cast("int32")
    source.mul_(10)
    assert converted.dtype == "int32"
    assert converted.device == source.device
    assert_tensor(converted, [[2, 3], [4, 5]])


@pytest.mark.parametrize("direction", ["from", "to"])
@pytest.mark.parametrize("offset", [-1, 4, 2**63-1])
def test_private_flat_copy_rejects_out_of_range_metadata(typed_tensor, operator_device, direction, offset):
    source = typed_tensor([7, 8], device=operator_device)
    destination = typed_tensor([0, 0, 0], device=operator_device)
    version = destination._storage_version
    with pytest.raises(IndexError, match="out of bounds"):
        if direction == "from":
            destination._copy_flat_range_from(offset, source)
        else:
            destination._copy_flat_range_to(offset, source)
    assert destination._storage_version == version
    assert destination.to("cpu").tolist() == [0, 0, 0]
    assert source.to("cpu").tolist() == [7, 8]


@pytest.mark.parametrize("direction", ["from", "to"])
def test_private_flat_copy_regular_remaining_range_and_endpoint(typed_tensor, operator_device, direction):
    values = typed_tensor([2, 3, 4], device=operator_device)
    shorter = typed_tensor([0, 0], device=operator_device)
    empty = typed_tensor([], device=operator_device)
    if direction == "from":
        version = values._storage_version
        values._copy_flat_range_from(1, shorter)
        assert values.to("cpu").tolist() == [2, 0, 0]
        assert values._storage_version == version + 1
        values._copy_flat_range_from(3, empty)
        assert values._storage_version == version + 2
        with pytest.raises(IndexError):
            values._copy_flat_range_from(2, shorter)
    else:
        version = shorter._storage_version
        values._copy_flat_range_to(1, shorter)
        assert shorter.to("cpu").tolist() == [3, 4]
        assert shorter._storage_version == version + 1
        version = empty._storage_version
        values._copy_flat_range_to(3, empty)
        assert empty._storage_version == version + 1
        with pytest.raises(IndexError):
            values._copy_flat_range_to(2, shorter)


def test_private_flat_copy_zero_offset_bumps_destination_once(typed_tensor, operator_device):
    source = typed_tensor([7, 8], device=operator_device)
    destination = typed_tensor([0, 0], device=operator_device)
    version = destination._storage_version
    source_version = source._storage_version
    destination._copy_flat_range_from(0, source)
    assert destination.to("cpu").tolist() == [7, 8]
    assert destination._storage_version == version + 1
    assert source._storage_version == source_version
    destination._copy_flat_range_to(0, source)
    assert source._storage_version == source_version + 1
    assert destination._storage_version == version + 1
