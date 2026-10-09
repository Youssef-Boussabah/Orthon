"""Test build provenance and usable Orthon devices, independent of PyTorch."""
PROFILES = ("cpu", "cuda", "cuda-nccl")


def build_profile(cuda_enabled, nccl_enabled):
    if nccl_enabled and not cuda_enabled:
        raise ValueError("NCCL support requires CUDA support")
    return "cuda-nccl" if nccl_enabled else "cuda" if cuda_enabled else "cpu"


def verify_profile(requested, cuda_enabled, nccl_enabled):
    if requested is not None and requested not in PROFILES:
        raise ValueError(
            "ORTHON_TEST_PROFILE must be cpu, cuda or cuda-nccl; got %r" % requested
        )
    actual = build_profile(cuda_enabled, nccl_enabled)
    if requested is not None and requested != actual:
        raise ValueError(
            "ORTHON_TEST_PROFILE=%s but loaded profile is %s; prepare the matching "
            "installed package or explicit PYTHONPATH stage" % (requested, actual)
        )
    return actual


def orthon_cuda_available(device="cuda:0"):
    """Probe usable CUDA; compiled but unavailable devices return False.

    Specifications keep their stricter cuda_device fixture: unexpected probe
    errors there must fail rather than become hardware skips.
    """
    import orthon
    try:
        orthon.zeros([1]).to(device)
        return True
    except RuntimeError:
        return False
