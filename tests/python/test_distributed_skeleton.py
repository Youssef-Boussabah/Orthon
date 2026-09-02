import json
import subprocess
import sys
import textwrap

import pytest

import orthon as ortho
import orthon.distributed as ddist


def teardown_function():
    ddist.destroy_process_group()


def test_distributed_imports():
    assert ddist.is_initialized() is False
    assert hasattr(ddist, "init_process_group")
    assert hasattr(ddist, "Work")


def test_nccl_backend_is_registered_but_requires_runtime_support():
    with pytest.raises((RuntimeError, ValueError)):
        ddist.init_process_group(backend="nccl", rank=0, world_size=1, store=None)
    assert ddist.is_initialized() is False


def test_single_rank_backend_lifecycle_and_noop_collectives():
    name = "_test_single_rank"
    ddist.register_backend(name, lambda **kwargs: ddist._SingleRankBackend())

    try:
        ddist.init_process_group(backend=name)
        assert ddist.is_initialized() is True
        assert ddist.get_rank() == 0
        assert ddist.get_world_size() == 1
        assert ddist.barrier() is None

        tensor = ortho.ones([3])
        before = tensor.tolist()
        assert ddist.all_reduce(tensor) is tensor
        assert tensor.tolist() == before
        assert ddist.broadcast(tensor, src=0) is tensor
        assert tensor.tolist() == before
        work = ddist.all_reduce(tensor, async_op=True)
        assert isinstance(work, ddist.Work)
        assert work.wait() is True
        assert work.wait() is True
        assert work.is_completed() is True
    finally:
        ddist.destroy_process_group()

    assert ddist.is_initialized() is False


def test_rank_config_from_env_parses_launcher_vars(monkeypatch):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    monkeypatch.setenv("ORTHON_DISTRIBUTED_BACKEND", "nccl")

    assert ddist.rank_config_from_env() == {
        "rank": 3,
        "world_size": 8,
        "local_rank": 1,
        "local_world_size": 4,
        "backend": "nccl",
    }


def test_init_process_group_uses_env_rank_world_size(monkeypatch):
    class _EnvBackend(ddist.Backend):
        def __init__(self, rank, world_size):
            self._rank = rank
            self._world_size = world_size

        @property
        def rank(self):
            return self._rank

        @property
        def world_size(self):
            return self._world_size

        def all_reduce(self, tensor, op="sum", async_op=False):
            return tensor

        def broadcast(self, tensor, src=0, async_op=False):
            return tensor

        def barrier(self):
            return None

        def shutdown(self):
            return None

    name = "_test_env_backend"
    ddist.register_backend(name, lambda **kwargs: _EnvBackend(**kwargs))
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("ORTHON_DISTRIBUTED_BACKEND", name)

    try:
        ddist.init_process_group()
        assert ddist.get_rank() == 1
        assert ddist.get_world_size() == 2
    finally:
        ddist.destroy_process_group()


def test_distributed_launcher_sets_child_env(tmp_path):
    script = tmp_path / "print_env.py"
    script.write_text(textwrap.dedent(
        """
        import json
        import os
        import pathlib
        import sys

        out_dir = pathlib.Path(sys.argv[1])
        rank = os.environ["RANK"]
        payload = {
            "rank": rank,
            "world_size": os.environ["WORLD_SIZE"],
            "local_rank": os.environ["LOCAL_RANK"],
            "local_world_size": os.environ["LOCAL_WORLD_SIZE"],
            "backend": os.environ.get("ORTHON_DISTRIBUTED_BACKEND"),
        }
        (out_dir / f"rank_{rank}.json").write_text(json.dumps(payload))
        """
    ))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "orthon.distributed.run",
            "--nproc_per_node",
            "2",
            "--backend",
            "local",
            str(script),
            str(tmp_path),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payloads = [
        json.loads((tmp_path / f"rank_{rank}.json").read_text())
        for rank in range(2)
    ]
    assert payloads == [
        {
            "rank": "0",
            "world_size": "2",
            "local_rank": "0",
            "local_world_size": "2",
            "backend": "local",
        },
        {
            "rank": "1",
            "world_size": "2",
            "local_rank": "1",
            "local_world_size": "2",
            "backend": "local",
        },
    ]


def test_distributed_launcher_reports_child_failure(tmp_path):
    script = tmp_path / "fail_rank_one.py"
    script.write_text(textwrap.dedent(
        """
        import os
        import sys

        if os.environ["RANK"] == "1":
            raise SystemExit(7)
        raise SystemExit(0)
        """
    ))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "orthon.distributed.run",
            "--nproc_per_node",
            "2",
            str(script),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode != 0
    assert "rank 1 exited with 7" in result.stderr
