import json
import os
import sys

import pytest

from .subprocesses import run_bounded


CHILD = '''import json, os, sys, time
from pathlib import Path
import orthon.distributed as dist
config = dist.rank_config_from_env()
record = dict(config=config, args=sys.argv[2:], initialized=dist.is_initialized(), pid=os.getpid())
Path(sys.argv[1], str(config["rank"]) + ".json").write_text(json.dumps(record))
if "--fail-rank-one" in sys.argv and config["rank"] == 1:
    # This case observes a failure after both workers have completed their
    # recording phase. The separate slow-peer case exercises immediate abort.
    deadline = time.monotonic() + 10
    while not all(Path(sys.argv[1], str(rank) + ".json").exists() for rank in range(config["world_size"])):
        if time.monotonic() >= deadline:
            raise RuntimeError("peers did not complete their recording phase")
        time.sleep(.01)
    raise SystemExit(7)
'''


@pytest.mark.parametrize("backend", [None, "local"], ids=["inherited-backend", "explicit-backend"])
def test_launcher_populates_rank_environment_and_preserves_literal_arguments(tmp_path, backend):
    script = tmp_path / "worker with spaces.py"
    script.write_text(CHILD)
    environment = dict(os.environ, RANK="99", WORLD_SIZE="99", LOCAL_RANK="99",
                       LOCAL_WORLD_SIZE="99", ORTHON_DISTRIBUTED_BACKEND="local")
    arguments = ["literal with spaces", "--literal-option", "value"]
    command = [sys.executable, "-m", "orthon.distributed.run", "--nproc_per_node", "2"]
    if backend:
        command += ["--backend", backend]
    result = run_bounded(command + [str(script), str(tmp_path), *arguments], env=environment)
    assert result.returncode == 0, result.stderr
    records = [json.loads((tmp_path / f"{rank}.json").read_text()) for rank in range(2)]
    assert records[0]["pid"] != records[1]["pid"]
    for rank, record in enumerate(records):
        assert record["config"] == dict(rank=rank, world_size=2, local_rank=rank,
                                         local_world_size=2, backend="local")
        assert record["args"] == arguments
        assert record["initialized"] is False


def test_launcher_propagates_completed_child_failure(tmp_path):
    script = tmp_path / "failure.py"
    script.write_text(CHILD)
    result = run_bounded([sys.executable, "-m", "orthon.distributed.run", "--nproc_per_node", "2",
                          "--backend", "local", str(script), str(tmp_path), "--fail-rank-one"])
    assert result.returncode == 7
    assert "rank 1" in result.stderr
    assert all((tmp_path / f"{rank}.json").exists() for rank in range(2))


@pytest.mark.parametrize("count", [0, -1])
def test_launcher_rejects_nonpositive_count_before_running_child(tmp_path, count):
    script = tmp_path / "must not execute.py"
    script.write_text("raise AssertionError('child executed')\n")
    result = run_bounded([sys.executable, "-m", "orthon.distributed.run", "--nproc_per_node",
                          str(count), str(script)])
    assert result.returncode == 2
    assert "child executed" not in result.stderr
