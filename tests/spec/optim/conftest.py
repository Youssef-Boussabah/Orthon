import pytest

from orthon.optim import Adam, SGD


@pytest.fixture(params=[
    pytest.param((SGD, "cpu", "float32"), id="sgd-cpu-float32"),
    pytest.param((SGD, "cpu", "float64"), id="sgd-cpu-float64"),
    pytest.param((Adam, "cpu", "float32"), id="adam-cpu-float32"),
    pytest.param((SGD, "cuda:0", "float32"), id="sgd-cuda-float32"),
    pytest.param((Adam, "cuda:0", "float32"), id="adam-cuda-float32"),
])
def optimizer_case(request):
    kind, device, dtype = request.param
    if device != "cpu":
        request.getfixturevalue("cuda_device")
    return kind, device, dtype


@pytest.fixture
def parameter(optimizer_case, typed_tensor):
    _, device, dtype = optimizer_case

    def create(values, *, requires_grad=True):
        return typed_tensor(values, dtype=dtype, device=device).requires_grad_(requires_grad)

    return create


@pytest.fixture
def optimizer(optimizer_case):
    kind, _, _ = optimizer_case

    def create(params, **kwargs):
        return kind(params, lr=kwargs.pop("lr", 0.25), **kwargs)

    return create


@pytest.fixture
def assign_gradient(typed_tensor):
    def assign(p, values):
        p.set_grad(typed_tensor(values, dtype=p.dtype, device=p.device))

    return assign


@pytest.fixture
def adam_parameter(operator_device, typed_tensor):
    def create(values):
        return typed_tensor(values, device=operator_device).requires_grad_(True)

    return create


@pytest.fixture
def copy_optimizer_state():
    def copy(state):
        result = dict(state)
        for name in ("m", "v"):
            if name in result:
                result[name] = [None if value is None else value.to(value.device)
                                for value in result[name]]
        return result

    return copy
