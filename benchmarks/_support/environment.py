"""Best-effort reproducibility metadata and framework completion."""
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys


def command(args, cwd=None):
    try:
        p = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=10)
        return p.stdout.strip() if p.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def cuda_available(device="cuda:0"):
    try:
        import orthon as o
        if not o.cuda_enabled():
            return False
        o.zeros([1]).to(device).to("cpu").item()
        return True
    except (ImportError, RuntimeError):
        return False


def synchronizer(device, framework="orthon"):
    if not str(device).startswith("cuda"):
        return lambda: None
    if framework == "torch":
        import torch
        return lambda: torch.cuda.synchronize(device)
    import orthon as o
    return o.cuda_synchronize


def cpu_identifier():
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                return line.split(':',1)[1].strip()
    except OSError:
        pass
    return platform.processor() or command(['uname','-m'])


def provenance():
    import orthon as o
    root = Path(__file__).resolve().parents[2]
    dirty = command(['git', 'status', '--porcelain'], root)
    try:
        version = importlib.metadata.version('orthon')
    except importlib.metadata.PackageNotFoundError:
        version = None
    try:
        import torch
        torch_info = dict(torch_version=torch.__version__, torch_cuda_version=torch.version.cuda,
                          torch_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                          torch_cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                          torch_float32_matmul_precision=torch.get_float32_matmul_precision())
    except ImportError:
        torch_info = dict(torch_version=None, torch_cuda_version=None,
                          torch_matmul_allow_tf32=None, torch_cudnn_allow_tf32=None,
                          torch_float32_matmul_precision=None)
    return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                git_commit=command(['git','rev-parse','HEAD'],root),
                git_dirty=None if dirty is None else bool(dirty),
                orthon_version=version, orthon_path=o.__file__, extension_path=o._C.__file__,
                python=sys.version, platform=platform.platform(), cpu=cpu_identifier(), logical_cpu_count=os.cpu_count(),
                cuda_enabled=bool(o.cuda_enabled()), nccl_enabled=bool(o.nccl_enabled()),
                gpu=command(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader']),
                cuda_toolkit=command(['nvcc','--version']), **torch_info)


def global_warmup(device, quick=False):
    """Fixed Orthon-only prelude to leave GPU idle clocks; never a sample."""
    if str(device).startswith('cuda'):
        import orthon as o
        a = o.ones([256,256]).to(device)
        for _ in range(8 if quick else 100):
            o.matmul(a,a)
        synchronizer(device)()


@contextmanager
def temporary_environment(values):
    prior = {name: os.environ.get(name) for name in values}
    try:
        for name, value in values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = str(value)
        yield
    finally:
        for name, value in prior.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
