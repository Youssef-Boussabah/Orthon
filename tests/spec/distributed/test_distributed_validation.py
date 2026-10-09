import numpy as np
import pytest

import orthon as ortho
import orthon.distributed as dist
from orthon.nn import Module, Parameter


ENVIRONMENT = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "ORTHON_DISTRIBUTED_BACKEND")


@pytest.fixture
def clean_environment(monkeypatch):
    for name in ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    assert not dist.is_initialized()
    yield
    dist.destroy_process_group()


@pytest.mark.parametrize("operation", ["rank", "world", "barrier", "reduce", "broadcast"])
def test_queries_and_collectives_require_initialization(clean_environment, operation):
    value = ortho.from_numpy(np.array([3., 7.], dtype=np.float32))
    functions = {"rank": dist.get_rank, "world": dist.get_world_size, "barrier": dist.barrier,
                 "reduce": lambda: dist.all_reduce(value), "broadcast": lambda: dist.broadcast(value)}
    with pytest.raises(RuntimeError):
        functions[operation]()
    assert value.tolist() == [3., 7.]
    assert not dist.is_initialized()
    assert dist.destroy_process_group() is None


def test_rank_environment_parser_handles_missing_empty_and_explicit_values():
    expected = dict(rank=None, world_size=None, local_rank=None, local_world_size=None, backend=None)
    assert dist.rank_config_from_env({}) == expected
    assert dist.rank_config_from_env({name: "" for name in ENVIRONMENT[:4]}) == expected
    assert dist.rank_config_from_env(dict(RANK="1", WORLD_SIZE="3", LOCAL_RANK="1",
         LOCAL_WORLD_SIZE="3", ORTHON_DISTRIBUTED_BACKEND="local")) == dict(
         rank=1, world_size=3, local_rank=1, local_world_size=3, backend="local")


@pytest.mark.parametrize("name", ENVIRONMENT[:4])
def test_malformed_rank_environment_rejects_initialization_without_state(clean_environment, monkeypatch, name):
    monkeypatch.setenv(name, "not-an-integer")
    with pytest.raises(ValueError):
        dist.rank_config_from_env()
    with pytest.raises(ValueError):
        dist.init_process_group(backend="local", rank=0, world_size=1, store=None)
    assert not dist.is_initialized()


@pytest.mark.parametrize("backend", [None, "unknown-spec-backend"])
def test_missing_or_unknown_backend_rejects_initialization(clean_environment, backend):
    with pytest.raises(ValueError):
        dist.init_process_group(backend=backend)
    assert not dist.is_initialized()


def test_local_initialization_requires_explicit_or_inferred_rank_world_and_store(clean_environment):
    with pytest.raises(TypeError):
        dist.init_process_group(backend="local")
    assert not dist.is_initialized()


@pytest.mark.parametrize("rank,world", [(-1, 2), (2, 2), (0, 0), (0, -1), (0, 1)])
def test_invalid_local_rank_world_or_missing_store_leaves_group_uninitialized(clean_environment, rank, world):
    with pytest.raises(ValueError):
        dist.init_process_group(backend="local", rank=rank, world_size=world, store=None)
    assert not dist.is_initialized()


@pytest.mark.parametrize("configuration", [{"bucket_size_mb": 0}, {"bucket_size_mb": -1},
    {"overlap_grad_sync": True}, {"overlap_grad_sync": True, "bucket_size_mb": 1, "sync_grads": False}])
def test_invalid_ddp_configuration_rejects_before_parameter_broadcast(clean_environment, configuration):
    model = Module()
    model.weight = Parameter(ortho.from_numpy(np.array([2., 5.], dtype=np.float32)))
    with pytest.raises(ValueError):
        dist.DDP(model, **configuration)
    assert model.weight.tolist() == [2., 5.]
    assert model.weight.grad is None


def test_local_collective_validation_preserves_values_and_allows_later_collectives(distributed_job):
    results = distributed_job("local-validation")
    for result in results:
        assert result["values"] == [3., 4., 6., 8.]
        assert result["shape"] == [2, 2]


@pytest.mark.parametrize("count", [0, -1])
def test_spawn_rejects_nonpositive_process_count_before_starting_workers(clean_environment, count):
    with pytest.raises(ValueError):
        dist.spawn(None, nprocs=count)
    assert not dist.is_initialized()


def test_worker_exception_propagates_after_other_workers_finish_and_cleanup(distributed_job):
    assert distributed_job("worker-error") == {"error": "RuntimeError", "worker_exception": "ValueError"}
    assert not dist.is_initialized()


def test_nccl_backend_reports_unavailable_build_without_initializing(clean_environment):
    if ortho.cuda_enabled() and ortho.nccl_enabled():
        pytest.skip("this contract requires a build without NCCL")
    with pytest.raises(RuntimeError):
        dist.init_process_group(backend="nccl", rank=0, world_size=1, device=0)
    assert not dist.is_initialized()


def test_direct_fp16_ddp_averaging_rejects_without_changing_parameters_or_gradients(distributed_job):
    assert distributed_job("low-precision-ddp") == ["RuntimeError", "RuntimeError"]
