"""Compatibility facade for the private, pickle-free v1 serialization service.

The same ordered logical state produces identical archive bytes. Model state
names are sorted by the service; user dictionary insertion order is preserved.
"""
from ._serialization.format import FORMAT_ID, FORMAT_VERSION
from ._serialization.service import SerializationService

_service = SerializationService()


def save_checkpoint(path, model=None, optimizer=None, metadata=None):
    """Synchronously snapshot and atomically replace a v1 checkpoint archive."""
    return _service.save_checkpoint(path, model, optimizer, metadata)


def load_checkpoint(path, model=None, optimizer=None, map_location=None):
    """Validate the whole archive, prepare recipients, restore and return metadata."""
    return _service.load_checkpoint(path, model, optimizer, map_location)


def save_state_dict(model, path):
    """Save an independent snapshot of the model's live state dictionary."""
    save_checkpoint(path, model=model)


def load_state_dict(model, path, map_location=None):
    """Restore model state through the same whole-file validation transaction."""
    load_checkpoint(path, model=model, map_location=map_location)
