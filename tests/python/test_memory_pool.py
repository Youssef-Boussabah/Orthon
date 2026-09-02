import gc

import pytest
import orthon


_UNLIMITED_CACHE = (1 << 64) - 1


def _reset_pool(enabled=True):
    gc.collect()
    orthon._memory_pool_set_enabled(True)
    orthon._memory_pool_set_max_cached_bytes(_UNLIMITED_CACHE)
    orthon._memory_pool_clear_cache()
    orthon._memory_pool_reset_scratch("cpu")
    orthon._memory_pool_reset_stats()
    orthon._memory_pool_set_enabled(enabled)


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

    stats = orthon._memory_pool_stats()
    assert stats["total_allocations"] >= 2
    assert stats["raw_allocations"] == 1
    assert stats["reused_allocations"] >= 1


def test_cpu_near_size_allocations_reuse_same_bin():
    _reset_pool(True)

    t = orthon.zeros([257])  # 1028 bytes -> 1088-byte bin.
    del t
    gc.collect()

    t = orthon.zeros([258])  # 1032 bytes -> same 1088-byte bin.
    del t
    gc.collect()

    stats = orthon._memory_pool_stats()
    assert stats["total_allocations"] == 2
    assert stats["raw_allocations"] == 1
    assert stats["reused_allocations"] == 1
    assert stats["current_bytes"] == 0


def test_memory_pool_evicts_cached_blocks_over_limit():
    _reset_pool(True)
    orthon._memory_pool_set_max_cached_bytes(4096)

    tensors = [orthon.zeros([1024]) for _ in range(3)]
    del tensors
    gc.collect()

    stats = orthon._memory_pool_stats()
    assert orthon._memory_pool_max_cached_bytes() == 4096
    assert stats["current_bytes"] == 0
    assert stats["cached_bytes"] <= 4096
    assert stats["cached_bytes"] == 4096


def test_memory_pool_lowering_limit_evicts_existing_cache():
    _reset_pool(True)

    tensors = [orthon.zeros([1024]) for _ in range(2)]
    del tensors
    gc.collect()
    assert orthon._memory_pool_stats()["cached_bytes"] == 8192

    orthon._memory_pool_set_max_cached_bytes(4096)

    stats = orthon._memory_pool_stats()
    assert orthon._memory_pool_max_cached_bytes() == 4096
    assert stats["current_bytes"] == 0
    assert stats["cached_bytes"] == 4096


def test_cpu_scratch_allocator_uses_bump_pointer_and_reset():
    _reset_pool(True)

    first = orthon._memory_pool_scratch_allocate(100)
    second = orthon._memory_pool_scratch_allocate(200)
    stats = orthon._memory_pool_scratch_stats("cpu")

    assert first != 0
    assert second == first + 128
    assert stats["reserved_bytes"] >= 1024 * 1024
    assert stats["used_bytes"] == 384
    assert stats["peak_used_bytes"] >= 384

    orthon._memory_pool_reset_scratch("cpu")
    third = orthon._memory_pool_scratch_allocate(64)
    stats = orthon._memory_pool_scratch_stats("cpu")

    assert third == first
    assert stats["used_bytes"] == 64
    assert orthon._memory_pool_stats()["total_allocations"] == 0


def test_cpu_scratch_allocator_grows_after_reset_for_large_request():
    _reset_pool(True)

    small = orthon._memory_pool_scratch_allocate(64)
    initial_reserved = orthon._memory_pool_scratch_stats("cpu")["reserved_bytes"]

    orthon._memory_pool_reset_scratch("cpu")
    large = orthon._memory_pool_scratch_allocate(initial_reserved + 64)
    stats = orthon._memory_pool_scratch_stats("cpu")

    assert small != 0
    assert large != 0
    assert stats["reserved_bytes"] >= initial_reserved + 64
    assert stats["used_bytes"] == initial_reserved + 64
    assert orthon._memory_pool_stats()["raw_allocations"] == 0


def test_memory_pool_disable_uses_raw_allocations():
    _reset_pool(False)

    for _ in range(3):
        t = orthon.zeros([512])
        del t
        gc.collect()

    stats = orthon._memory_pool_stats()
    assert stats["total_allocations"] == 3
    assert stats["raw_allocations"] == 3
    assert stats["reused_allocations"] == 0


def test_repeated_allocations_do_not_double_free_or_crash():
    _reset_pool(True)

    for _ in range(100):
        tensors = [orthon.zeros([128]) for _ in range(8)]
        del tensors
        gc.collect()

    stats = orthon._memory_pool_stats()
    assert stats["reused_allocations"] > 0


@pytest.mark.skipif(not orthon.cuda_enabled(), reason="CUDA is not available")
def test_cuda_repeated_tensor_allocations_reuse_memory():
    _reset_pool(True)

    t = orthon.zeros([1024]).to("cuda")
    del t
    gc.collect()

    t = orthon.zeros([1024]).to("cuda")
    del t
    gc.collect()

    stats = orthon._memory_pool_stats()
    assert stats["reused_allocations"] >= 1
