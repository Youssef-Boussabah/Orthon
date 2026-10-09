import argparse
import os
import subprocess
import sys
import tempfile
import time


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="python -m orthon.distributed.run",
        description="Launch a single-node Orthon distributed job.",
    )
    parser.add_argument("--nproc_per_node", type=int, required=True)
    parser.add_argument("--backend", default=None)
    parser.add_argument("training_script")
    parser.add_argument("training_script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.nproc_per_node < 1:
        parser.error("--nproc_per_node must be >= 1")
    return args


def _child_env(base_env, rank, world_size, backend, nccl_store_dir=None):
    env = dict(base_env)
    env["RANK"] = str(rank)
    env["WORLD_SIZE"] = str(world_size)
    env["LOCAL_RANK"] = str(rank)
    env["LOCAL_WORLD_SIZE"] = str(world_size)
    if backend is not None:
        env["ORTHON_DISTRIBUTED_BACKEND"] = backend
    if nccl_store_dir is not None:
        env["ORTHON_NCCL_STORE_DIR"] = nccl_store_dir
    return env


def launch(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    processes = []
    command_tail = [args.training_script] + args.training_script_args
    store_context = (
        tempfile.TemporaryDirectory(prefix="orthon_nccl_")
        if args.backend == "nccl"
        else None
    )
    try:
        nccl_store_dir = None if store_context is None else store_context.__enter__()
        for rank in range(args.nproc_per_node):
            env = _child_env(
                os.environ,
                rank,
                args.nproc_per_node,
                args.backend,
                nccl_store_dir=nccl_store_dir,
            )
            process = subprocess.Popen([sys.executable] + command_tail, env=env)
            processes.append((rank, process))

        failures = []
        while True:
            failures = [(rank, p.returncode) for rank, p in processes
                        if p.poll() not in (None, 0)]
            if failures or all(p.poll() is not None for _, p in processes):
                break
            time.sleep(0.05)
        if failures:
            details = ", ".join(
                f"rank {rank} exited with {code}" for rank, code in failures
            )
            print(f"orthon.distributed.run failed: {details}", file=sys.stderr)
            return failures[0][1] or 1
        return 0
    finally:
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
        for _, process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
        if store_context is not None:
            store_context.__exit__(None, None, None)


def main():
    raise SystemExit(launch())


if __name__ == "__main__":
    main()
