import json
import sys

import pytest

import orthon as ortho

from .subprocesses import run_bounded


@pytest.fixture
def distributed_job():
    def run(scenario, *, nprocs=2, **options):
        request = dict(scenario=scenario, nprocs=nprocs, options=options)
        result = run_bounded([sys.executable, "-m", "tests.spec.distributed.workloads"],
                             payload=json.dumps(request))
        assert result.returncode == 0, result.stdout + result.stderr
        records = [line[len("ORTHON_SPEC_RESULT="):] for line in result.stdout.splitlines()
                   if line.startswith("ORTHON_SPEC_RESULT=")]
        assert len(records) == 1, result.stdout + result.stderr
        return json.loads(records[0])
    return run


@pytest.fixture(scope="session")
def nccl_two_devices():
    if not ortho.cuda_enabled() or not ortho.nccl_enabled():
        pytest.skip("distributed NCCL specifications require a CUDA+NCCL build")
    for index in range(2):
        try:
            ortho.zeros([1]).to(f"cuda:{index}")
        except RuntimeError as error:
            message = str(error).lower()
            if any(reason in message for reason in ("no cuda-capable device", "driver version is insufficient",
                       "initialization error", "unsupported display driver", "invalid device ordinal")):
                pytest.skip(f"distributed NCCL specifications need two usable devices: {error}")
            raise
    return ["cuda:0", "cuda:1"]
