"""What an installed Orthon provides.

These run against whatever is importable - a wheel, an editable install or a
PYTHONPATH tree - so they are the checks that survive packaging. The
source-tree half of the contract is in test_build_architecture.py.
"""
import importlib
import importlib.metadata
import importlib.util
import hashlib
import pkgutil
import re

import pytest

import orthon


def test_orthon_imports_and_reports_a_build_flavour():
    assert isinstance(orthon.cuda_enabled(), bool)


def test_the_extension_module_is_private_and_lives_under_orthon():
    assert orthon._C.__name__ == "orthon._C"
    assert orthon.Tensor.__module__ == "orthon._C"


def test_retired_private_surfaces_are_absent_from_package_and_extension():
    from orthon.nn import Module
    diag = importlib.import_module("orthon._diagnostics")
    assert callable(diag.memory_pool_stats)
    assert not any(n.startswith(("_matmul_", "_memory_pool_", "cuda_conv2d_profile_")) for n in dir(orthon))
    assert not hasattr(orthon, "_scalar_like")
    assert not hasattr(orthon._C, "_scalar_like")
    assert not hasattr(Module, "_replace_parameter")
    assert not hasattr(Module(), "_replace_parameter")


@pytest.mark.parametrize("name", [
    "Tensor", "Device", "DeviceType", "DType",
    "zeros", "ones", "empty", "full", "from_numpy",
    "add", "sub", "mul", "div", "neg", "matmul",
    "reshape", "transpose", "contiguous",
    "sum", "mean", "max", "argmax",
    "relu", "sigmoid", "tanh", "gelu", "leaky_relu", "exp", "sqrt",
    "softmax", "log_softmax", "cross_entropy", "nll_loss",
    "conv2d", "maxpool2d", "dropout", "embedding", "cast", "to",
    "manual_seed", "no_grad", "is_grad_enabled", "set_grad_enabled",
    "save_checkpoint", "load_checkpoint",
    "save_state_dict", "load_state_dict",
    "cuda_enabled", "cuda_synchronize",
    "nn", "optim", "amp", "training",
])
def test_public_api_surface(name):
    assert hasattr(orthon, name), "orthon.%s is missing" % name


@pytest.mark.parametrize("name", ["nn", "optim", "amp", "functional",
                                  "distributed", "training", "checkpoint"])
def test_subpackages_import_under_the_orthon_namespace(name):
    module = importlib.import_module("orthon." + name)
    assert module.__name__ == "orthon." + name


def test_every_submodule_is_addressed_under_the_orthon_package():
    """One package, one namespace: nothing ships beside or above `orthon`."""
    found = [info.name for info in
             pkgutil.walk_packages(orthon.__path__, "orthon.")]
    assert found, "expected at least one submodule"
    for name in found:
        assert name.startswith("orthon."), name


def test_checkpoint_format_identity_is_independent_of_package_branding():
    """The on-disk identifier deliberately does not name the package.

    Keeping the two independent is what lets the package be renamed, vendored
    or forked without any checkpoint migration.
    """
    from orthon.checkpoint import FORMAT_ID, FORMAT_VERSION
    assert FORMAT_ID == "ml-runtime-checkpoint"
    assert FORMAT_VERSION == 1
    assert "orthon" not in FORMAT_ID.lower()


# Installed-distribution metadata. Skipped on a bare PYTHONPATH tree, where
# there is no distribution to describe.
def _distribution():
    try:
        return importlib.metadata.metadata("orthon")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("orthon is importable but not installed as a distribution")


def test_distribution_metadata_is_accurate():
    meta = _distribution()
    assert meta["Name"] == "orthon"
    assert meta["Version"] == "1.0.0rc1"
    assert ">=3.10" in meta["Requires-Python"]
    assert any(req.startswith("numpy") for req in meta.get_all("Requires-Dist"))


def test_distribution_carries_the_readme_as_its_long_description():
    """`readme = "README.md"` must actually reach the built metadata."""
    meta = _distribution()
    assert meta["Description-Content-Type"] == "text/markdown"
    body = meta.get_payload() or meta["Description"] or ""
    assert "# Orthon" in body
    assert "C++17/CUDA tensor and autograd runtime" in body


def test_distribution_contains_complete_license_notices():
    meta = _distribution()
    assert set(meta.get_all("License-File", [])) == {"LICENSE", "THIRD_PARTY_NOTICES.md"}
    distribution = importlib.metadata.distribution("orthon")
    payloads = {}
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        files = [p for p in distribution.files or []
                 if str(p).endswith(".dist-info/licenses/" + name)]
        assert len(files) == 1, (name, files)
        payloads[name] = distribution.locate_file(files[0]).read_bytes()
    # The existing Orthon MIT license must remain complete and unchanged.
    assert hashlib.sha256(payloads["LICENSE"]).hexdigest() == (
        "67cac3de78ceb89e72503402ce81cb4c47a1ed61115cc01f5590dfafcd266000")
    blocks = re.findall(r"^```text\n(.*?)^```$",
                        payloads["THIRD_PARTY_NOTICES.md"].decode(), re.M | re.S)
    assert len(blocks) == 1
    assert hashlib.sha256(blocks[0].encode()).hexdigest() == (
        "83965b843b98f670d3a85bd041ed4b372c8ec50d7b4a5995a83ac697ba675dcb")


def test_distribution_declares_no_unearned_metadata():
    """License and stewardship are declared without inventing exclusive authorship.

    `.get()` rather than `meta[field]`: an absent header returns None from the
    subscript only by deprecated behaviour that later raises KeyError.
    """
    meta = _distribution()
    assert meta.get("License-Expression") == "MIT"
    assert "LICENSE" in meta.get_all("License-File", [])
    assert meta.get("Maintainer") == "Youssef Boussabah"
    for field in ("Author", "Author-email", "Maintainer-email",
                  "Home-page", "License", "Project-URL"):
        value = meta.get(field)
        assert not value, "%s is set to %r" % (field, value)
