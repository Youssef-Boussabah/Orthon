import numpy as np
import pytest

import orthon as ortho


def test_view_strides_and_offsets_describe_logical_coordinates(tensor, assert_tensor):
    source = tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    transposed = source.transpose(-1, -2)
    narrowed = transposed.slice(0, 1, 4).slice(1, 1, 3)
    assert transposed.strides == tuple(reversed(source.strides))
    assert transposed.storage_offset == source.storage_offset
    assert narrowed.strides == transposed.strides
    assert narrowed.storage_offset == source.storage_offset + sum(source.strides)
    assert source.is_contiguous is True
    assert narrowed.is_contiguous is False
    assert_tensor(narrowed, [[6, 10], [7, 11], [8, 12]])


@pytest.mark.parametrize("name", ["sum", "mean", "max", "argmax"])
@pytest.mark.parametrize("layout", ["transpose", "slice"])
def test_reductions_read_non_contiguous_inputs_in_logical_order(
    operator_profile, typed_tensor, assert_output, name, layout,
):
    device, dtype = operator_profile
    values = np.array([[4, 1, 9], [2, 8, 3]])
    source = typed_tensor(values, dtype=dtype, device=device)
    if layout == "transpose":
        source, values = source.transpose(0, 1), values.T
    else:
        source, values = source.slice(1, 1, 3), values[:, 1:3]
    assert_output(getattr(ortho, name)(source, dim=-1), getattr(np, name)(values, axis=-1),
                  dtype="int64" if name == "argmax" else dtype, device=device)


def test_binary_operators_combine_strided_coordinates_with_broadcasting(
    operator_profile, typed_tensor, binary_contract, assert_output,
):
    device, dtype = operator_profile
    public, syntax = binary_contract
    values = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    source = typed_tensor(values, dtype=dtype, device=device).slice(1, 1, 4).transpose(0, 1)
    other = typed_tensor([[2], [4], [8]], dtype=dtype, device=device)
    assert_output(public(source, other), syntax(values[:, 1:4].T, np.array([[2], [4], [8]])),
                  dtype=dtype, device=device)


@pytest.mark.parametrize("batched", [False, True])
def test_matmul_accepts_strided_operands_without_changing_their_values(
    operator_profile, typed_tensor, assert_output, batched,
):
    device, dtype = operator_profile
    left_values = np.array([[1, 2], [3, 4], [5, 6]])
    right_values = np.array([[7, 8, 9], [10, 11, 12]])
    if batched:
        left_values = np.stack([left_values, left_values + 1])
        right_values = np.stack([right_values, right_values - 1])
    left = typed_tensor(left_values, dtype=dtype, device=device).transpose(-1, -2)
    right = typed_tensor(right_values, dtype=dtype, device=device).transpose(-1, -2)
    expected_left, expected_right = left_values.swapaxes(-1, -2), right_values.swapaxes(-1, -2)
    assert_output(ortho.matmul(left, right), expected_left @ expected_right, dtype=dtype, device=device)
    assert_output(left, expected_left, dtype=dtype, device=device)
    assert_output(right, expected_right, dtype=dtype, device=device)


def test_cpu_layout_materialization_is_available_for_every_storage_dtype(
    storage_dtype, typed_tensor, assert_output,
):
    values = np.array([[0, 1, 0], [1, 0, 1]])
    source = typed_tensor(values, dtype=storage_dtype).transpose(0, 1)
    materialized = source.contiguous()
    assert materialized.is_contiguous is True
    assert_output(materialized, values.T, dtype=storage_dtype, device="cpu")
    assert_output(source.cast("float32"), values.T, dtype="float32", device="cpu")


def test_cuda_strided_materialization_has_a_narrower_dtype_boundary(
    cuda_device, storage_dtype, typed_tensor, assert_output,
):
    values = np.array([[0, 1, 0], [1, 0, 1]])
    source = typed_tensor(values, dtype=storage_dtype, device=cuda_device).transpose(0, 1)
    if storage_dtype in ("float32", "float16", "bfloat16"):
        output = source.contiguous()
        assert output.is_contiguous is True
        assert_output(output, values.T, dtype=storage_dtype, device=cuda_device)
    else:
        with pytest.raises(RuntimeError):
            source.contiguous()


@pytest.mark.parametrize("layout", ["transpose", "step"])
def test_numpy_import_requires_c_contiguity_instead_of_implicitly_packing(layout):
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    array = values.T if layout == "transpose" else values[:, ::2]
    with pytest.raises(ValueError):
        ortho.from_numpy(array)
