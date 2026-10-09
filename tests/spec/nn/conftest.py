import pytest

import orthon as ortho
from orthon.nn import Module, Parameter


@pytest.fixture
def module_factory(tensor, tensor_case):
    def create():
        model = Module()
        model.weight = Parameter(tensor([[1, 2], [3, 4]]))
        model.branch = Module()
        model.branch.bias = Parameter(tensor([5, 6]))
        model.branch.register_buffer("running", tensor([7, 8]))
        model.branch.register_buffer("cache", tensor([9, 10]), persistent=False)
        model.branch.register_buffer("optional", None)
        counter = ortho.full([1], 2, dtype="int64")
        model.branch.register_buffer("counter", counter.to(tensor_case[0]))
        model.scratch = tensor([11, 12])
        return model

    return create
