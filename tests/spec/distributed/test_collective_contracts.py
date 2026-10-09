import numpy as np
import pytest


@pytest.mark.parametrize("dtype", ["float32", "float64", "float16", "int32", "int64", "uint8", "int8"])
def test_local_sum_and_each_broadcast_source_preserve_tensor_metadata(distributed_job, dtype):
    results = distributed_job("collectives", dtype=dtype)
    for rank_result in results:
        for round_result in rank_result:
            total = round_result["summed"]
            assert total["values"] == [3., 6., 9., 12.]
            assert total["dtype"] == dtype and total["shape"] == [2, 2] and total["device"] == "cpu"
            for source, broadcast in enumerate(round_result["broadcasts"]):
                assert broadcast["values"] == [1 + 10 * source, 2 + 10 * source, 3 + 10 * source, 4 + 10 * source]
                assert broadcast["dtype"] == dtype and broadcast["shape"] == [2, 2]


@pytest.mark.parametrize("shape", [[], [1], [0, 2]])
@pytest.mark.parametrize("world", [1, 2])
def test_scalar_singleton_and_empty_local_collectives_keep_logical_shapes(distributed_job, shape, world):
    results = distributed_job("collectives", nprocs=world, shape=shape)
    expected = (np.arange(1, np.prod(shape, dtype=int) + 1) * world * (world + 1) / 2).tolist()
    for result in results:
        for round_result in result:
            assert round_result["summed"]["shape"] == shape
            assert round_result["summed"]["values"] == expected
            assert all(value["shape"] == shape for value in round_result["broadcasts"])


def test_boolean_local_broadcast_copies_source_values_without_numerical_sum_assumptions(distributed_job):
    results = distributed_job("bool-broadcast")
    for result in results:
        assert [entry["values"] for entry in result] == [[True, False], [False, True]]
        assert all(entry["dtype"] == "bool" and entry["shape"] == [2] for entry in result)


def test_barriers_after_collectives_observe_every_ranks_arrival_in_each_round(distributed_job):
    results = distributed_job("barrier", nprocs=3)
    assert results == [[[True] * 3] * 3] * 3
