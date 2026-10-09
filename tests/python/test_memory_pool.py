import orthon._diagnostics as diag
from tests.support.capabilities import orthon_cuda_available
import gc

import pytest
import orthon


_UNLIMITED_CACHE = (1 << 64) - 1


def _reset_pool(enabled=True):
    gc.collect()
    diag.memory_pool_set_enabled(True)
    diag.memory_pool_set_max_cached_bytes(_UNLIMITED_CACHE)
    diag.memory_pool_clear_cache()
    diag.memory_pool_reset_scratch("cpu")
    diag.memory_pool_reset_stats()
    diag.memory_pool_set_enabled(enabled)
    return diag.memory_pool_stats()["current_bytes"]


def teardown_function(_):
    _reset_pool(True)


def test_cpu_repeated_tensor_allocations_reuse_memory():
    _reset_pool(True)

    t = orthon.zeros([1024])
    del t
    gc.collect()

    t = orthon.zeros([1024])
    del t
    gc.collect()

    stats = diag.memory_pool_stats()
    assert stats["total_allocations"] >= 2
    assert stats["raw_allocations"] == 1
    assert stats["reused_allocations"] >= 1


def test_cpu_near_size_allocations_reuse_same_bin():
    baseline_current = _reset_pool(True)

    t = orthon.zeros([257])  # 1028 bytes -> 1088-byte bin.
    del t
    gc.collect()

    t = orthon.zeros([258])  # 1032 bytes -> same 1088-byte bin.
    del t
    gc.collect()

    stats = diag.memory_pool_stats()
    assert stats["total_allocations"] == 2
    assert stats["raw_allocations"] == 1
    assert stats["reused_allocations"] == 1
    # Reset preserves unrelated allocations already live in this process.
    assert stats["current_bytes"] == baseline_current


def test_memory_pool_evicts_cached_blocks_over_limit():
    baseline_current = _reset_pool(True)
    diag.memory_pool_set_max_cached_bytes(4096)

    tensors = [orthon.zeros([1024]) for _ in range(3)]
    del tensors
    gc.collect()

    stats = diag.memory_pool_stats()
    assert diag.memory_pool_max_cached_bytes() == 4096
    assert stats["current_bytes"] == baseline_current
    assert stats["cached_bytes"] <= 4096
    assert stats["cached_bytes"] == 4096


def test_memory_pool_lowering_limit_evicts_existing_cache():
    baseline_current = _reset_pool(True)

    tensors = [orthon.zeros([1024]) for _ in range(2)]
    del tensors
    gc.collect()
    assert diag.memory_pool_stats()["cached_bytes"] == 8192

    diag.memory_pool_set_max_cached_bytes(4096)

    stats = diag.memory_pool_stats()
    assert diag.memory_pool_max_cached_bytes() == 4096
    assert stats["current_bytes"] == baseline_current
    assert stats["cached_bytes"] == 4096


def test_cpu_scratch_allocator_uses_bump_pointer_and_reset():
    _reset_pool(True)

    first = diag.memory_pool_scratch_allocate(100)
    second = diag.memory_pool_scratch_allocate(200)
    stats = diag.memory_pool_scratch_stats("cpu")

    assert first != 0
    assert second == first + 128
    assert stats["reserved_bytes"] >= 1024 * 1024
    assert stats["used_bytes"] == 384
    assert stats["peak_used_bytes"] >= 384

    diag.memory_pool_reset_scratch("cpu")
    third = diag.memory_pool_scratch_allocate(64)
    stats = diag.memory_pool_scratch_stats("cpu")

    assert third == first
    assert stats["used_bytes"] == 64
    assert diag.memory_pool_stats()["total_allocations"] == 0


def test_cpu_scratch_allocator_grows_after_reset_for_large_request():
    _reset_pool(True)

    small = diag.memory_pool_scratch_allocate(64)
    initial_reserved = diag.memory_pool_scratch_stats("cpu")["reserved_bytes"]

    diag.memory_pool_reset_scratch("cpu")
    large = diag.memory_pool_scratch_allocate(initial_reserved + 64)
    stats = diag.memory_pool_scratch_stats("cpu")

    assert small != 0
    assert large != 0
    assert stats["reserved_bytes"] >= initial_reserved + 64
    assert stats["used_bytes"] == initial_reserved + 64
    assert diag.memory_pool_stats()["raw_allocations"] == 0


def test_memory_pool_disable_uses_raw_allocations():
    _reset_pool(False)

    for _ in range(3):
        t = orthon.zeros([512])
        del t
        gc.collect()

    stats = diag.memory_pool_stats()
    assert stats["total_allocations"] == 3
    assert stats["raw_allocations"] == 3
    assert stats["reused_allocations"] == 0


def test_repeated_allocations_do_not_double_free_or_crash():
    _reset_pool(True)

    for _ in range(100):
        tensors = [orthon.zeros([128]) for _ in range(8)]
        del tensors
        gc.collect()

    stats = diag.memory_pool_stats()
    assert stats["reused_allocations"] > 0


def test_stats_reset_preserves_live_gauges_and_release_cannot_underflow():
    _reset_pool(True)
    before = diag.memory_pool_stats()
    tensor = orthon.zeros([4])
    live = diag.memory_pool_stats()
    assert live["current_bytes"] == before["current_bytes"] + 16
    assert live["reserved_bytes"] == before["reserved_bytes"] + 64
    diag.memory_pool_reset_stats()
    reset = diag.memory_pool_stats()
    assert reset["current_bytes"] == live["current_bytes"]
    assert reset["reserved_bytes"] == live["reserved_bytes"]
    assert reset["peak_bytes"] == reset["current_bytes"]
    assert reset["total_allocations"] == 0
    del tensor
    gc.collect()
    after = diag.memory_pool_stats()
    assert after["current_bytes"] == before["current_bytes"]
    assert after["cached_bytes"] == 64
    assert after["reserved_bytes"] == live["reserved_bytes"]


def test_stats_reset_preserves_cache_and_clear_releases_only_cached_capacity():
    _reset_pool(True)
    tensor = orthon.zeros([4])
    del tensor
    gc.collect()
    before = diag.memory_pool_stats()
    assert before["cached_bytes"] == 64
    diag.memory_pool_reset_stats()
    reset = diag.memory_pool_stats()
    assert reset["current_bytes"] == before["current_bytes"]
    assert reset["cached_bytes"] == before["cached_bytes"]
    assert reset["reserved_bytes"] == before["reserved_bytes"]
    diag.memory_pool_clear_cache()
    cleared = diag.memory_pool_stats()
    assert cleared["current_bytes"] == before["current_bytes"]
    assert cleared["cached_bytes"] == 0
    assert cleared["reserved_bytes"] == before["reserved_bytes"] - 64


@pytest.mark.skipif(not orthon_cuda_available(), reason="CUDA is not available")
def test_cuda_repeated_tensor_allocations_reuse_memory():
    _reset_pool(True)

    t = orthon.zeros([1024]).to("cuda")
    del t
    gc.collect()

    t = orthon.zeros([1024]).to("cuda")
    del t
    gc.collect()

    stats = diag.memory_pool_stats()
    assert stats["reused_allocations"] >= 1
