import numpy as np
import pytest

import orthon as ortho


@pytest.fixture(scope="session")
def cuda_device():
    if not ortho.cuda_enabled():
        pytest.skip("spec CUDA profile requires a CUDA-enabled Orthon build")
    try:
        ortho.zeros([1]).to("cuda:0")
    except RuntimeError as error:
        message = str(error).lower()
        if any(reason in message for reason in (
            "no cuda-capable device",
            "cuda driver version is insufficient",
            "initialization error",
            "unsupported display driver / cuda driver combination",
        )):
            pytest.skip(f"spec CUDA profile has no usable device: {error}")
        raise
    return "cuda:0"


@pytest.fixture(scope="session", params=[
    pytest.param(("cpu", "float32"), id="cpu-float32"),
    pytest.param(("cpu", "float64"), id="cpu-float64"),
    pytest.param(("cuda:0", "float32"), id="cuda-float32"),
])
def tensor_case(request):
    device, dtype = request.param
    if device != "cpu":
        request.getfixturevalue("cuda_device")
    return device, dtype


@pytest.fixture
def tensor(tensor_case):
    device, dtype = tensor_case

    def create(values, *, requires_grad=False):
        result = ortho.from_numpy(np.array(values, dtype=dtype, order="C"))
        if device != "cpu":
            result = result.to(device)
        return result.requires_grad_(requires_grad)

    return create


@pytest.fixture
def assert_tensor():
    def check(actual, expected):
        expected = np.asarray(expected)
        assert tuple(actual.shape) == expected.shape
        tolerance = 1e-12 if actual.dtype == "float64" else 1e-6
        assert actual.tolist() == pytest.approx(
            expected.reshape(-1).tolist(), rel=tolerance, abs=tolerance,
        )

    return check


@pytest.fixture
def assert_grad(assert_tensor):
    def check(leaf, expected):
        assert leaf.grad is not None
        assert_tensor(leaf.grad, expected)
        assert leaf.grad.dtype == leaf.dtype
        assert leaf.grad.device == leaf.device
        assert leaf.grad.requires_grad is False
        assert leaf.grad.is_leaf is True

    return check


@pytest.fixture(params=["float32", "float64", "float16", "bfloat16",
                        "int32", "int64", "uint8", "int8", "bool"])
def storage_dtype(request):
    return request.param


@pytest.fixture(params=["cpu", "cuda:0"], ids=["cpu", "cuda"])
def operator_device(request):
    if request.param != "cpu":
        request.getfixturevalue("cuda_device")
    return request.param


@pytest.fixture(params=[
    pytest.param(("cpu", "float32"), id="cpu-float32"),
    pytest.param(("cpu", "float64"), id="cpu-float64"),
    pytest.param(("cuda:0", "float32"), id="cuda-float32"),
    pytest.param(("cuda:0", "float16"), id="cuda-float16"),
    pytest.param(("cuda:0", "bfloat16"), id="cuda-bfloat16"),
])
def operator_profile(request):
    device, dtype = request.param
    if device != "cpu":
        request.getfixturevalue("cuda_device")
    return device, dtype


@pytest.fixture
def typed_tensor():
    def create(values, *, dtype="float32", device="cpu"):
        array = np.array(values, dtype="float32" if dtype == "bfloat16" else dtype,
                         order="C")
        result = (ortho.from_numpy(array, dtype="bfloat16") if dtype == "bfloat16"
                  else ortho.from_numpy(array))
        return result if device == "cpu" else result.to(device)

    return create


@pytest.fixture
def assert_output():
    def check(actual, expected, *, dtype, device):
        expected = np.asarray(expected)
        assert actual.shape == expected.shape
        assert actual.dtype == dtype
        assert actual.device == ortho.Device(device)
        if dtype in ("int32", "int64", "uint8", "int8", "bool"):
            assert actual.tolist() == expected.reshape(-1).tolist()
        else:
            tolerance = {"float64": 1e-12, "float32": 2e-6,
                         "float16": 5e-3, "bfloat16": 4e-2}[dtype]
            assert actual.tolist() == pytest.approx(
                expected.reshape(-1).tolist(), rel=tolerance, abs=tolerance,
            )

    return check


@pytest.fixture(params=["add", "sub", "mul", "div"])
def binary_contract(request):
    import operator

    name = request.param
    syntax = {"add": operator.add, "sub": operator.sub,
              "mul": operator.mul, "div": operator.truediv}[name]
    return getattr(ortho, name), syntax
