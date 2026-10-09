"""Public CUDA ordering, ephemeral operand and host boundary regressions."""
import orthon._diagnostics as diag
import numpy as np
import pytest
import orthon as ortho

pytestmark = pytest.mark.skipif(not ortho.cuda_enabled(), reason='CUDA build required')


@pytest.fixture(autouse=True)
def usable_cuda_device(cuda_device):
    # Keep the existing CPU-build skip message, then apply the strict spec probe.
    return cuda_device


def tensor(values, dtype='float32'):
    value = ortho.from_numpy(np.asarray(values, dtype=np.float32)).to('cuda:0')
    return ortho.cast(value, dtype) if dtype != 'float32' else value


@pytest.mark.parametrize('dtype', ['float32', 'float16', 'bfloat16'])
def test_pipelined_custom_cublas_custom_and_final_host_read(dtype):
    x = tensor([[1, 2], [3, 4]], dtype)
    identity = tensor(np.eye(2), dtype)
    outputs = []
    for _ in range(40):
        value = ortho.relu(ortho.matmul(ortho.add(x, x), identity))
        outputs.append(ortho.sum(value))
    assert [value.item() for value in outputs] == pytest.approx([20] * 40)


@pytest.mark.parametrize('dtype', ['float32', 'float16', 'bfloat16'])
def test_temporary_views_casts_reductions_and_reuse_without_manual_sync(dtype):
    x = tensor(np.arange(24).reshape(4, 6), dtype)
    outputs = []
    for _ in range(40):
        packed = ortho.cast(x.transpose(0, 1), 'float32')
        outputs.append(ortho.sum(ortho.add(packed, packed)))
        del packed
    assert [value.item() for value in outputs] == pytest.approx([552] * 40)


@pytest.mark.parametrize('dtype', ['float32', 'float16', 'bfloat16'])
def test_reduction_scratch_reuse_between_allocating_ops(dtype):
    x = tensor(np.ones(262145), dtype)
    outputs = []
    for _ in range(16):
        outputs.extend([ortho.mean(x), ortho.max(x), ortho.mean(ortho.add(x, x))])
    assert [value.item() for value in outputs] == pytest.approx([1, 1, 2] * 16)


def test_host_scalar_write_then_arithmetic_reads_new_value():
    x = tensor([1, 2, 3])
    x.set_item([1], 9)
    assert ortho.add(x, x).tolist() == pytest.approx([2, 18, 6])


def test_same_device_copy_is_independent_and_ordered():
    x = tensor([1, 2, 3])
    result = ortho.add(x, x).to('cuda:0')
    x.set_item([0], 99)
    assert result.to('cpu').tolist() == pytest.approx([2, 4, 6])


def test_cpu_source_snapshot_survives_immediate_mutation():
    x = ortho.from_numpy(np.array([1, 2, 3], dtype=np.float32))
    result = x.to('cuda:0')
    x.set_item([0], 99)
    assert result.tolist() == pytest.approx([1, 2, 3])


def test_finite_and_amp_bool_are_host_boundaries():
    x = tensor([1, np.inf])
    assert not ortho.isfinite_all(ortho.add(x, x))
    output, finite = ortho._amp_unscale_to_float32_and_check(x, .5)
    assert not finite
    assert output.tolist() == [0.5, float('inf')]


def test_autograd_view_temporaries_are_safe_through_final_gradient_read():
    x = tensor(np.arange(12).reshape(3, 4))
    x.requires_grad_(True)
    for _ in range(20):
        loss = ortho.sum(ortho.mul(x.transpose(0, 1), x.transpose(0, 1)))
        loss.backward()
    assert np.asarray(x.grad.tolist()).reshape(3, 4) == pytest.approx(np.arange(12).reshape(3, 4) * 40)


def test_scratch_and_pool_stats_preserve_pending_capacity():
    diag.memory_pool_clear_cache()
    x = tensor(np.ones(10000))
    value = ortho.sum(x)
    stats = diag.memory_pool_scratch_stats('cuda:0')
    assert stats['used_bytes'] == 0
    assert stats['reserved_bytes'] > 0
    assert value.item() == pytest.approx(10000)
    diag.memory_pool_clear_cache()
    assert diag.memory_pool_stats()['cached_bytes'] == 0
