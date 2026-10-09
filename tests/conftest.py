"""Cross-suite pytest ownership and explicit loaded-package provenance."""
import importlib
import os
import platform
import pytest
from tests.support.capabilities import PROFILES, verify_profile
from tests.support.layers import PRIMARY_LAYERS, primary_layer_for_path

def pytest_configure(config):
    requested = os.environ.get("ORTHON_TEST_PROFILE")
    # Validate the requested spelling even if no package has been prepared.
    if requested is not None and requested not in PROFILES:
        raise pytest.UsageError("ORTHON_TEST_PROFILE must be cpu, cuda or cuda-nccl; got %r" % requested)
    try:
        orthon = importlib.import_module("orthon")
    except ModuleNotFoundError as error:
        if error.name in ("orthon", "orthon._C"):
            raise pytest.UsageError("Orthon is unavailable; install a built package or supply an explicit PYTHONPATH stage") from error
        raise
    cuda, nccl = orthon.cuda_enabled(), orthon.nccl_enabled()
    try:
        actual = verify_profile(requested, cuda, nccl)
    except ValueError as error:
        raise pytest.UsageError(str(error) + "; package=" + str(orthon.__file__) + "; extension=" + str(orthon._C.__file__)) from error
    config._orthon_test_provenance = [
        "Orthon package: " + str(orthon.__file__),
        "Orthon extension: " + str(orthon._C.__file__),
        "Orthon profile: %s (cuda_enabled=%s, nccl_enabled=%s, expected=%s)" % (actual, cuda, nccl, requested or "unset"),
        "Orthon validation Python: " + platform.python_version(),
    ]

def pytest_report_header(config):
    return config._orthon_test_provenance

def pytest_collection_modifyitems(items):
    for item in items:
        try:
            expected = primary_layer_for_path(item.path)
        except ValueError as error:
            raise pytest.UsageError(str(error)) from error
        primary = [m.name for m in item.iter_markers() if m.name in PRIMARY_LAYERS]
        if not primary:
            item.add_marker(getattr(pytest.mark, expected))
            primary = [expected]
        if primary != [expected]:
            raise pytest.UsageError("%s must have exactly one primary layer %s; found %s" % (item.nodeid, expected, primary))
