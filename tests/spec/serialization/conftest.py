import json
import zipfile

import pytest

import orthon as ortho

from .workload import RegressionModel, advance, make_optimizer, value_tree


@pytest.fixture(params=[
    pytest.param(("SGD", "cpu", "float32"), id="sgd-cpu-float32"),
    pytest.param(("SGD", "cpu", "float64"), id="sgd-cpu-float64"),
    pytest.param(("Adam", "cpu", "float32"), id="adam-cpu-float32"),
    pytest.param(("SGD", "cuda:0", "float32"), id="sgd-cuda-float32"),
    pytest.param(("Adam", "cuda:0", "float32"), id="adam-cuda-float32"),
])
def resume_case(request):
    name, device, dtype = request.param
    if device != "cpu":
        request.getfixturevalue("cuda_device")
    return name, device, dtype


@pytest.fixture
def training_pair(resume_case):
    name, device, dtype = resume_case

    def create(*, lr=0.125):
        model = RegressionModel(dtype, device)
        return model, make_optimizer(model, name, lr=lr)

    return create


class CheckpointArchive:
    def __init__(self, path):
        with zipfile.ZipFile(path) as archive:
            self.manifest = json.loads(archive.read("manifest.json"))
            self.payloads = {name: archive.read(name) for name in archive.namelist()
                             if name != "manifest.json"}

    def write(self, path, *, manifest_bytes=None, omit_manifest=False, compressed=()):
        with zipfile.ZipFile(path, "w") as archive:
            if not omit_manifest:
                archive.writestr("manifest.json", manifest_bytes if manifest_bytes is not None
                                 else json.dumps(self.manifest).encode("utf-8"))
            for name, payload in self.payloads.items():
                archive.writestr(name, payload, compress_type=(zipfile.ZIP_DEFLATED
                                 if name in compressed else zipfile.ZIP_STORED))
        return path

    def records(self, section):
        result = []

        def visit(node):
            if isinstance(node, dict):
                if node.get("k") == "tensor":
                    result.append(node)
                else:
                    for child in node.values():
                        visit(child)
            elif isinstance(node, list):
                for child in node:
                    visit(child)

        visit(self.manifest["sections"][section])
        return result


@pytest.fixture
def checkpoint_archive():
    return CheckpointArchive


@pytest.fixture
def assert_destination_unchanged():
    def capture(model, optimizer):
        state = value_tree(model.state_dict())
        tensors = dict(model.state_dict())
        parameters = list(model.parameters())
        gradients = [None if p.grad is None else value_tree(p.grad) for p in parameters]
        flags = [(p.requires_grad, p.is_leaf) for p in parameters]
        modes = model.training, model.head.training
        progress = value_tree(optimizer.state_dict())

        def check():
            assert value_tree(model.state_dict()) == state
            assert all(model.state_dict()[name] is old for name, old in tensors.items())
            assert [None if p.grad is None else value_tree(p.grad) for p in parameters] == gradients
            assert [(p.requires_grad, p.is_leaf) for p in parameters] == flags
            assert (model.training, model.head.training) == modes
            assert value_tree(optimizer.state_dict()) == progress

        return check

    return capture


@pytest.fixture
def saved_training_checkpoint(tmp_path):
    model = RegressionModel()
    optimizer = make_optimizer(model, "Adam")
    advance(model, optimizer, 1)
    path = tmp_path / "valid.ckpt"
    ortho.save_checkpoint(path, model=model, optimizer=optimizer,
                          metadata={"step": 1, "late": ortho.full([2], 7.)})
    return path


@pytest.fixture
def rejection_target():
    model = RegressionModel()
    optimizer = make_optimizer(model, "Adam", lr=0.25)
    advance(model, optimizer, 1)
    model.weight.mul_(2.)
    model.eval()
    return model, optimizer
