import pytest
import orthon as ortho
from orthon import Device, DeviceType

cpu = Device(DeviceType.CPU, 0)


def _cuda_available():
    try:
        ortho.zeros([1]).to("cuda")
        return True
    except RuntimeError:
        return False


no_cuda = pytest.mark.skipif(_cuda_available(), reason="CUDA is available")


def test_to_cpu_copies_values():
    t = ortho.full([2, 3], 5.0)
    c = t.to(cpu)
    assert c.tolist() == pytest.approx([5.0] * 6)


def test_to_cpu_independent_copy():
    t = ortho.full([3], 1.0)
    c = t.to(cpu)
    t.fill_(99.0)
    assert c.tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_to_str_cpu():
    t = ortho.full([2, 2], 3.0)
    c = t.to("cpu")
    assert c.tolist() == pytest.approx([3.0] * 4)


def test_to_device_object():
    t = ortho.ones([4])
    c = t.to(cpu)
    assert c.shape == (4,)
    assert c.tolist() == pytest.approx([1.0] * 4)


def test_to_preserves_shape():
    t = ortho.zeros([3, 4])
    c = t.to(cpu)
    assert c.shape == (3, 4)


def test_to_contiguous_result():
    t = ortho.ones([3, 3])
    c = t.to(cpu)
    assert c.is_contiguous is True


def test_to_noncontiguous_becomes_contiguous():
    t = ortho.full([2, 3], 1.0)
    tr = t.transpose(0, 1)
    assert tr.is_contiguous is False
    c = tr.to(cpu)
    assert c.is_contiguous is True
    assert c.shape == (3, 2)
    assert c.tolist() == pytest.approx([1.0] * 6)


@no_cuda
def test_to_cuda_raises():
    t = ortho.zeros([2, 2])
    with pytest.raises(RuntimeError):
        t.to("cuda")


def test_to_invalid_device_string():
    t = ortho.zeros([2, 2])
    with pytest.raises(Exception):
        t.to("tpu")


def test_module_level_to_accepts_device_object():
    t = ortho.full([2], 4.0)
    c = ortho.to(t, cpu)
    assert c.tolist() == pytest.approx([4.0, 4.0])


def test_module_level_to_accepts_device_string():
    t = ortho.full([2], 4.0)
    c = ortho.to(t, "cpu")
    assert c.tolist() == pytest.approx([4.0, 4.0])
