import os


_BACKEND_ENV = "ORTHON_DISTRIBUTED_BACKEND"


def _optional_int(environ, name):
    value = environ.get(name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def rank_config_from_env(environ=None):
    """Return distributed rank settings from launcher-style env vars."""
    if environ is None:
        environ = os.environ
    return {
        "rank": _optional_int(environ, "RANK"),
        "world_size": _optional_int(environ, "WORLD_SIZE"),
        "local_rank": _optional_int(environ, "LOCAL_RANK"),
        "local_world_size": _optional_int(environ, "LOCAL_WORLD_SIZE"),
        "backend": environ.get(_BACKEND_ENV),
    }


def apply_env_defaults(backend, kwargs, environ=None):
    config = rank_config_from_env(environ)
    if backend is None:
        backend = config["backend"]
    if backend is None:
        raise ValueError(
            "distributed backend must be provided explicitly or via "
            f"{_BACKEND_ENV}"
        )

    filled = dict(kwargs)
    if filled.get("rank") is None and config["rank"] is not None:
        filled["rank"] = config["rank"]
    if filled.get("world_size") is None and config["world_size"] is not None:
        filled["world_size"] = config["world_size"]
    if (
        backend == "nccl"
        and filled.get("device") is None
        and config["local_rank"] is not None
    ):
        filled["device"] = config["local_rank"]
    return backend, filled


__all__ = ["apply_env_defaults", "rank_config_from_env"]
