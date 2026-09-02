import pytest
import orthon as ortho
from orthon import Device, DeviceType


def _cuda_available():
    try:
        ortho.zeros([1]).to("cuda")
        return True
    except RuntimeError:
        return False


no_cuda = pytest.mark.skipif(_cuda_available(), reason="CUDA is available")


# Device construction.

def test_default_device_is_cpu():
    d = Device()
    assert d.is_cpu()
    assert not d.is_cuda()
    assert d.index == 0


def test_explicit_cpu_device():
    d = Device(DeviceType.CPU, 0)
    assert d.is_cpu()
    assert d.type == DeviceType.CPU


def test_explicit_cuda_device():
    d = Device(DeviceType.CUDA, 0)
    assert d.is_cuda()
    assert not d.is_cpu()
    assert d.index == 0


def test_device_equality():
    assert Device(DeviceType.CPU, 0) == Device(DeviceType.CPU, 0)
    assert Device(DeviceType.CUDA, 1) == Device(DeviceType.CUDA, 1)
    assert Device(DeviceType.CPU, 0) != Device(DeviceType.CUDA, 0)


def test_device_repr_cpu():
    d = Device(DeviceType.CPU, 0)
    assert repr(d) == "device('cpu')"


def test_device_repr_cuda():
    d = Device(DeviceType.CUDA, 0)
    assert repr(d) == "device('cuda:0')"


def test_device_repr_cuda_index():
    d = Device(DeviceType.CUDA, 2)
    assert repr(d) == "device('cuda:2')"


# Tensor.device property.

def test_tensor_device_defaults_to_cpu():
    t = ortho.zeros([3, 4])
    assert t.device.is_cpu()


def test_tensor_device_type():
    t = ortho.ones([2])
    assert t.device.type == DeviceType.CPU
    assert t.device.index == 0


def test_tensor_device_after_cpu_copy():
    t = ortho.full([2, 3], 7.0)
    c = t.to("cpu")
    assert c.device.is_cpu()


# CUDA unavailable in non-CUDA builds.

@no_cuda
def test_cuda_tensor_raises_runtime_error():
    t = ortho.zeros([2, 2])
    with pytest.raises(RuntimeError):
        t.to(Device(DeviceType.CUDA, 0))


@no_cuda
def test_cuda_string_raises_runtime_error():
    t = ortho.zeros([2, 2])
    with pytest.raises(RuntimeError):
        t.to("cuda")


@no_cuda
def test_cuda_indexed_string_raises_runtime_error():
    t = ortho.zeros([2, 2])
    with pytest.raises(RuntimeError):
        t.to("cuda:0")


def test_invalid_device_string_raises():
    t = ortho.zeros([2, 2])
    with pytest.raises(Exception):
        t.to("tpu")


# Strict device-string parsing.

def test_device_string_cuda_index_parses():
    d = Device("cuda:1")
    assert d.is_cuda()
    assert d.index == 1


def test_device_string_leading_zero_accepted():
    # "cuda:01" is ordinary decimal notation for index 1.
    assert Device("cuda:01").index == 1


@pytest.mark.parametrize("spec", [
    "cuda:",       # no index
    "cuda:-1",     # negative index
    "cuda:+1",     # sign prefix
    "cuda:1x",     # trailing garbage after the digits
    "cuda: 1",     # embedded whitespace
    "cuda:01x",    # garbage after leading-zero digits
    "cuda:one",    # non-numeric index
    " cuda",       # leading whitespace
    "cuda:0 ",     # trailing whitespace
    "CUDA",        # wrong case
    "gpu",         # unknown device name
])
def test_malformed_device_strings_rejected(spec):
    with pytest.raises(ValueError):
        Device(spec)


def test_malformed_device_string_rejected_by_tensor_to():
    t = ortho.zeros([2])
    with pytest.raises(ValueError):
        t.to("cuda:1x")


# Equality / hash contract.

def test_device_equal_values_hash_equal():
    assert Device("cpu") == Device("cpu")
    assert hash(Device("cpu")) == hash(Device("cpu"))
    assert Device("cuda:0") == Device("cuda:0")
    assert hash(Device("cuda:0")) == hash(Device("cuda:0"))
    assert Device("cpu") != Device("cuda:0")


def test_device_compare_with_non_device_is_false_not_error():
    assert (Device("cpu") == "cpu") is False
    assert (Device("cpu") != "cpu") is True
    assert (Device("cpu") == None) is False  # noqa: E711
    assert (Device("cpu") == 0) is False


def test_device_usable_as_dict_key():
    table = {Device("cpu"): "host", Device("cuda:0"): "gpu0"}
    assert table[Device("cpu")] == "host"
    assert table[Device("cuda:0")] == "gpu0"


# Immutability: Device hashes by value, so its fields must not be assignable
# from Python — mutating a key already inside a dict would change its hash.

def test_device_index_is_read_only():
    d = Device("cuda:0")
    with pytest.raises(AttributeError):
        d.index = 1
    assert d.index == 0


def test_device_type_is_read_only():
    d = Device("cuda:0")
    with pytest.raises(AttributeError):
        d.type = DeviceType.CPU
    assert d.type == DeviceType.CUDA


def test_device_hash_stable_in_dict_after_failed_mutation():
    d = Device(DeviceType.CUDA, 0)
    table = {d: "entry"}
    with pytest.raises(AttributeError):
        d.index = 1
    assert table[Device(DeviceType.CUDA, 0)] == "entry"


# Explicit Device(DeviceType, index) validation: cpu index must be 0, cuda
# index must be non-negative. Whether a cuda ordinal physically exists is
# not checked at construction time.

def test_explicit_cpu_nonzero_index_rejected():
    with pytest.raises(ValueError):
        Device(DeviceType.CPU, 1)


def test_explicit_cpu_negative_index_rejected():
    with pytest.raises(ValueError):
        Device(DeviceType.CPU, -1)


def test_explicit_cuda_negative_index_rejected():
    with pytest.raises(ValueError):
        Device(DeviceType.CUDA, -1)


def test_explicit_large_cuda_ordinal_is_syntactically_valid():
    assert Device(DeviceType.CUDA, 9999).index == 9999
