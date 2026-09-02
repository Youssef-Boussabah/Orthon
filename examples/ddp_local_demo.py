"""Tiny CPU-only DDP-lite demo using the local multiprocessing backend.

This is a correctness-oriented example, not a production distributed training
recipe. It uses explicit gradient synchronization between backward() and the
optimizer step.
"""

import numpy as np

import orthon as ortho
import orthon.distributed as ddist
from orthon.distributed import DistributedDataParallel as DDP
from orthon.nn import Linear
from orthon.optim import SGD


STEPS = 3
LR = 0.05


def fixed_model():
    model = Linear(2, 1)
    params = list(model.parameters())
    params[0].fill_(0.25)
    params[1].fill_(0.1)
    return model


def shard(rank):
    xs = [
        np.asarray([[1.0, 2.0]], dtype=np.float32),
        np.asarray([[3.0, 4.0]], dtype=np.float32),
    ]
    target_negs = [
        np.asarray([[-1.5]], dtype=np.float32),
        np.asarray([[-3.5]], dtype=np.float32),
    ]
    return ortho.from_numpy(xs[rank]), ortho.from_numpy(target_negs[rank])


def flatten_params(model):
    values = []
    for parameter in model.parameters():
        values.extend(float(value) for value in parameter.tolist())
    return values


def worker(rank):
    model = fixed_model()
    ddp_model = DDP(model)
    optimizer = SGD(ddp_model.parameters(), lr=LR)
    x, target_neg = shard(rank)
    losses = []

    for _ in range(STEPS):
        pred = ddp_model(x)
        diff = pred + target_neg
        loss = ortho.mean(diff * diff)
        losses.append(float(loss.item()))

        loss.backward()
        ddp_model.all_reduce_gradients()
        optimizer.step()
        optimizer.zero_grad()

    return {
        "rank": rank,
        "losses": losses,
        "final_params": flatten_params(model),
    }


def main():
    results = ddist.spawn(worker, nprocs=2)
    for result in results:
        losses = ", ".join(f"{value:.6f}" for value in result["losses"])
        params = ", ".join(f"{value:.6f}" for value in result["final_params"])
        print(f"rank {result['rank']}: losses=[{losses}] final_params=[{params}]")


if __name__ == "__main__":
    main()
