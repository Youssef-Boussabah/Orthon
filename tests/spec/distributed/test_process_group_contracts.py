import pytest

import orthon.distributed as dist


@pytest.mark.parametrize("world", [1, 2])
@pytest.mark.parametrize("explicit", [False, True], ids=["environment", "explicit-overrides-environment"])
def test_real_local_groups_can_be_destroyed_and_reinitialized(distributed_job, world, explicit):
    results = distributed_job("lifecycle", nprocs=world, explicit=explicit)
    assert results == [[{"rank": rank, "world": world}] * 2 for rank in range(world)]
    assert dist.is_initialized() is False


def test_spawn_assigns_distinct_processes_ranks_and_propagates_arguments(distributed_job):
    results = distributed_job("spawn", nprocs=3, argument={"values": [2, 5], "label": "literal input"})
    assert len({entry["pid"] for entry in results}) == 3
    for rank, result in enumerate(results):
        assert result["rank"] == result["queried_rank"] == rank
        assert result["world"] == 3 and result["initialized"] is True
        assert result["argument"] == {"values": [2, 5], "label": "literal input"}
    assert dist.is_initialized() is False


def test_repeated_spawn_calls_do_not_inherit_previous_group_state(distributed_job):
    first = distributed_job("spawn", argument="first")
    second = distributed_job("spawn", nprocs=1, argument="second")
    assert [entry["world"] for entry in first] == [2, 2]
    assert second[0]["rank"] == 0 and second[0]["world"] == 1
    assert second[0]["argument"] == "second"
    assert dist.is_initialized() is False
