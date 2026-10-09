_active = None


def set_active(group):
    global _active
    _active = group


def get_active():
    if _active is None:
        raise RuntimeError("distributed process group is not initialized")
    return _active


def clear_active():
    global _active
    _active = None


def is_initialized():
    return _active is not None


__all__ = [
    "set_active",
    "get_active",
    "clear_active",
    "is_initialized",
]
