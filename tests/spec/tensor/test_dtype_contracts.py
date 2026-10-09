import numpy as np
import pytest

import orthon as ortho


CORE_OPERATORS = ["add", "sub", "mul", "div", "neg", "relu", "sigmoid", "tanh",
                  "leaky_relu", "gelu", "exp", "sqrt", "reciprocal",
                  "sum", "mean", "max", "argmax", "matmul"]
DTYPES = ["float32", "float64", "float16", "bfloat16", "int32", "int64", "uint8", "int8", "bool"]


@pytest.mark.parametrize("factory", ["empty", "zeros", "ones", "full"])
@pytest.mark.parametrize("shape", [(), (2, 3), (2, 0, 3)])
def test_factory_metadata_and_defined_fill_values_cover_every_storage_dtype(
    storage_dtype, assert_output, factory, shape,
):
    kwargs = {"dtype": storage_dtype}
    if factory == "full":
        kwargs["value"] = 2
    output = getattr(ortho, factory)(list(shape), **kwargs)
    assert output.shape == shape
    assert output.dtype == storage_dtype
    assert output.device == ortho.Device("cpu")
    assert output.numel == int(np.prod(shape))
    if factory != "empty":
        fill = {"zeros": 0, "ones": 1, "full": 2}[factory]
        if storage_dtype == "bool":
            fill = bool(fill)
        assert_output(output, np.full(shape, fill), dtype=storage_dtype, device="cpu")


@pytest.mark.parametrize("factory", ["empty", "zeros", "ones", "full"])
def test_allocating_factories_default_to_cpu_float32(factory):
    output = getattr(ortho, factory)([2], **({"value": 1} if factory == "full" else {}))
    assert output.dtype == "float32"
    assert output.device == ortho.Device("cpu")


def test_numpy_import_preserves_native_dtype_rank_and_logical_values(
    storage_dtype, typed_tensor, assert_output,
):
    values = np.array([[0, 1], [2, 3]], dtype=bool if storage_dtype == "bool" else int)
    assert_output(typed_tensor(values, dtype=storage_dtype), values,
                  dtype=storage_dtype, device="cpu")


@pytest.mark.parametrize("source_dtype", DTYPES)
def test_cpu_casts_connect_all_storage_dtypes_without_changing_shape_or_device(
    storage_dtype, typed_tensor, assert_output, source_dtype,
):
    source = typed_tensor([[0, 1], [1, 0]], dtype=source_dtype)
    for output in (source.cast(storage_dtype), ortho.cast(source, storage_dtype)):
        assert_output(output, [[0, 1], [1, 0]], dtype=storage_dtype, device="cpu")


@pytest.mark.parametrize("target", ["int32", "int64", "int8"])
def test_finite_in_range_float_to_integer_cast_truncates_toward_zero(typed_tensor, assert_output, target):
    assert_output(typed_tensor([-2.75, -.5, .5, 2.75]).cast(target), [-2, 0, 0, 2],
                  dtype=target, device="cpu")


def test_bool_cast_means_zero_versus_nonzero(typed_tensor, assert_output):
    assert_output(typed_tensor([-2, 0, 3]).cast("bool"), [True, False, True],
                  dtype="bool", device="cpu")


@pytest.mark.parametrize("dtype,values,expected", [
    ("float16", [2049, 2051], [2048, 2052]),
    ("bfloat16", [1.00390625, 1.01171875], [1, 1.015625]),
])
def test_low_precision_rounding_agrees_across_creation_import_and_cast(
    typed_tensor, assert_output, dtype, values, expected,
):
    imported = typed_tensor(values, dtype=dtype)
    converted = typed_tensor(values).cast(dtype)
    assert_output(imported, expected, dtype=dtype, device="cpu")
    assert_output(converted, expected, dtype=dtype, device="cpu")
    for value, rounded in zip(values, expected):
        assert ortho.full([], value, dtype=dtype).item() == rounded
    assert imported.tolist() == expected
    assert converted.tolist() == expected


@pytest.mark.parametrize("name", CORE_OPERATORS)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "int32", "int64", "uint8", "int8", "bool"])
def test_storage_support_does_not_imply_cpu_arithmetic_support(typed_tensor, name, dtype):
    source = typed_tensor([[1, 2], [3, 4]], dtype=dtype)
    with pytest.raises(RuntimeError):
        if name in ("add", "sub", "mul", "div", "matmul"):
            getattr(ortho, name)(source, source)
        else:
            getattr(ortho, name)(source)


@pytest.mark.parametrize("name", CORE_OPERATORS)
@pytest.mark.parametrize("dtype", ["float64", "int32", "int64", "uint8", "int8", "bool"])
def test_cuda_arithmetic_rejects_storage_only_dtypes(cuda_device, typed_tensor, name, dtype):
    source = typed_tensor([[1, 2], [3, 4]], dtype=dtype, device=cuda_device)
    with pytest.raises(RuntimeError):
        if name in ("add", "sub", "mul", "div", "matmul"):
            getattr(ortho, name)(source, source)
        else:
            getattr(ortho, name)(source)


@pytest.mark.parametrize("name", ["add", "sub", "mul", "div", "matmul"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("other_dtype", [dtype for dtype in DTYPES if dtype != "float32"])
def test_tensor_operands_are_not_implicitly_promoted_to_a_common_dtype(typed_tensor, name, reverse, other_dtype):
    left = typed_tensor([[1, 2], [3, 4]], dtype="float32")
    right = typed_tensor([[1, 2], [3, 4]], dtype=other_dtype)
    with pytest.raises(RuntimeError):
        getattr(ortho, name)(right, left) if reverse else getattr(ortho, name)(left, right)


@pytest.mark.parametrize("name", ["add", "sub", "mul", "div", "matmul"])
@pytest.mark.parametrize("left_dtype,right_dtype", [
    ("float32", "float16"), ("float16", "bfloat16"), ("bfloat16", "float32"),
])
@pytest.mark.parametrize("reverse", [False, True])
def test_supported_cuda_compute_dtypes_still_do_not_mix(
    cuda_device, typed_tensor, name, left_dtype, right_dtype, reverse,
):
    left = typed_tensor([[1, 2], [3, 4]], dtype=left_dtype, device=cuda_device)
    right = typed_tensor([[1, 2], [3, 4]], dtype=right_dtype, device=cuda_device)
    with pytest.raises(RuntimeError):
        getattr(ortho, name)(right, left) if reverse else getattr(ortho, name)(left, right)


@pytest.mark.parametrize("source_dtype", DTYPES)
def test_contiguous_cuda_casts_connect_all_storage_dtypes(
    cuda_device, storage_dtype, typed_tensor, assert_output, source_dtype,
):
    source = typed_tensor([[0, 1], [1, 0]], dtype=source_dtype, device=cuda_device)
    assert_output(source.cast(storage_dtype), [[0, 1], [1, 0]],
                  dtype=storage_dtype, device=cuda_device)


@pytest.mark.parametrize("dtype", ["int16", "uint32", "uint64", "complex64", "object"])
def test_numpy_import_rejects_unexposed_array_dtypes(dtype):
    with pytest.raises(TypeError):
        ortho.from_numpy(np.array([1, 2], dtype=dtype))


@pytest.mark.parametrize("dtype", ["float32", "float64", "float16", "int32", "bool"])
def test_numpy_explicit_dtype_override_is_only_for_bfloat16(dtype):
    with pytest.raises(TypeError):
        ortho.from_numpy(np.array([1, 2], dtype="float32"), dtype=dtype)


@pytest.mark.parametrize("dtype", ["float64", "float16", "int32"])
def test_bfloat16_import_requires_float32_input(dtype):
    with pytest.raises(TypeError):
        ortho.from_numpy(np.array([1, 2], dtype=dtype), dtype="bfloat16")


@pytest.mark.parametrize("operation", ["empty", "zeros", "ones", "full", "cast"])
def test_unknown_dtype_names_are_validation_errors(typed_tensor, operation):
    with pytest.raises(ValueError):
        if operation == "cast":
            typed_tensor([1]).cast("float128")
        else:
            kwargs = {"dtype": "float128"}
            if operation == "full":
                kwargs["value"] = 1
            getattr(ortho, operation)([1], **kwargs)
