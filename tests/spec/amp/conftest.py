import pytest

from .workload import AffineModel


@pytest.fixture(params=[
    pytest.param(("cpu", "float32"), id="cpu-float32"),
    pytest.param(("cpu", "float16"), id="cpu-float16-storage"),
    pytest.param(("cpu", "bfloat16"), id="cpu-bfloat16-storage"),
    pytest.param(("cuda:0", "float32"), id="cuda-float32"),
    pytest.param(("cuda:0", "float16"), id="cuda-float16"),
    pytest.param(("cuda:0", "bfloat16"), id="cuda-bfloat16"),
])
def amp_storage_case(request):
    device, dtype = request.param
    if device != "cpu":
        request.getfixturevalue("cuda_device")
    return device, dtype


@pytest.fixture(params=[
    pytest.param(("cpu", "float32"), id="cpu-float32"),
    pytest.param(("cuda:0", "float16"), id="cuda-float16"),
    pytest.param(("cuda:0", "bfloat16"), id="cuda-bfloat16"),
])
def amp_training_case(request):
    device, dtype = request.param
    if device != "cpu":
        request.getfixturevalue("cuda_device")
    return device, dtype


@pytest.fixture
def amp_model(amp_storage_case, typed_tensor):
    device, dtype = amp_storage_case
    return AffineModel(lambda values: typed_tensor(values, dtype=dtype, device=device))


@pytest.fixture
def training_model(amp_training_case, typed_tensor):
    device, dtype = amp_training_case
    return AffineModel(lambda values: typed_tensor(values, dtype=dtype, device=device))


@pytest.fixture
def model_gradient(typed_tensor):
    def assign(parameter, values, *, strided=False):
        gradient = typed_tensor(values, dtype=parameter.dtype, device=parameter.device)
        if strided:
            gradient = gradient.transpose(0, 1)
        parameter.set_grad(gradient)
        return gradient
    return assign
