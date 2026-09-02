import math

import pytest

import orthon as ortho
import orthon.distributed as ddist
from orthon.distributed import DistributedDataParallel as DDP
from orthon.distributed.local_backend import create_store
from orthon.nn import Linear
from orthon.optim import SGD


try:
    ortho.zeros([1]).to("cuda")
    CUDA_AVAILABLE = True
except Exception:
    CUDA_AVAILABLE = False


requires_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")


@requires_cuda
def test_ddp_world_size_one_cuda_training_smoke():
    manager, store = create_store()
    try:
        ddist.init_process_group(
            backend="local",
            rank=0,
            world_size=1,
            store=store,
        )

        model = Linear(2, 1)
        for parameter in model.parameters():
            parameter.fill_(0.25)
        model.to("cuda")

        optimizer = SGD(model.parameters(), lr=0.01)
        ddp_model = DDP(model)
        x = ortho.full([1, 2], 2.0).to("cuda")
        target_neg = ortho.full([1, 1], -1.0).to("cuda")

        pred = ddp_model(x)
        diff = pred + target_neg
        loss = ortho.mean(diff * diff)
        loss.backward()
        ddp_model.all_reduce_gradients()
        optimizer.step()
        optimizer.zero_grad()

        for parameter in model.parameters():
            values = parameter.to("cpu").tolist()
            assert all(math.isfinite(float(value)) for value in values)
    finally:
        ddist.destroy_process_group()
        manager.shutdown()
