"""Whole-file snapshot/decode orchestration through state-owner protocols."""
import orthon as ortho
from .archive import ArchiveCodec
from .codec import ValueCodec
from .format import FORMAT_ID, FORMAT_VERSION, _fail
from .transaction import RestoreTransaction


def _normalize_map_location(spec, api):
    """Normalize a string or Device once so every state owner compares exact Devices."""
    if spec is None or isinstance(spec, ortho.Device):
        return spec
    if isinstance(spec, str):
        try:
            return ortho.Device(spec)
        except Exception as exc:
            raise ValueError("%s: unknown map_location %r (%s)"
                             % (api, spec, exc))
    raise TypeError(
        "%s: map_location must be a device string or Device, got %r"
        % (api, type(spec)))


def _require_state_dict(obj, api, what):
    if not hasattr(obj, "state_dict"):
        raise TypeError("%s: %s must expose state_dict()" % (api, what))


def _encode_model(encoder, model, api):
    """Sort model names for deterministic archives; preserve in-memory ordering."""
    _require_state_dict(model, api, "model")
    state = model.state_dict()
    items = []
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, ortho.Tensor):
            raise TypeError(
                "%s: model state entry %r must be a Tensor, got %r"
                % (api, name, type(tensor)))
        items.append([name, encoder.encode(tensor, "model.%s" % name)])
    return {"k": "dict", "v": items}


class SerializationService:
    def save_checkpoint(self, path, model=None, optimizer=None, metadata=None):
        target = ArchiveCodec.fspath(path, "save_checkpoint")
        encoder = ValueCodec.encoder()
        model_section = None if model is None else _encode_model(encoder, model, "save_checkpoint")
        optimizer_section = None
        if optimizer is not None:
            _require_state_dict(optimizer, "save_checkpoint", "optimizer")
            optimizer_section = encoder.encode(optimizer.state_dict(), "optimizer")
        metadata_section = encoder.encode(dict(metadata or {}), "metadata")
        manifest = {
            "format": FORMAT_ID,
            "format_version": FORMAT_VERSION,
            "sections": {"model": model_section, "optimizer": optimizer_section,
                         "metadata": metadata_section},
        }
        ArchiveCodec.write(target, manifest, encoder.payloads)

    def load_checkpoint(self, path, model=None, optimizer=None, map_location=None):
        target = _normalize_map_location(map_location, "load_checkpoint")
        with ArchiveCodec.open(path, "load_checkpoint") as archive:
            sections = ArchiveCodec.read_manifest(archive)
            absent = {name for name in ("model", "optimizer") if sections[name] is None}
            decoder = ValueCodec.decoder(lambda name, size, where:
                                   ArchiveCodec.read_payload(archive, name, size, where))
            restored = {name: None if sections[name] is None and name != "metadata" else
                        decoder.decode(sections[name], name)
                        for name in ("model", "optimizer", "metadata")}
            model_state = restored["model"]
            if sections["model"] is not None and (not isinstance(model_state, dict) or
                    not all(isinstance(k, str) and isinstance(v, ortho.Tensor)
                            for k, v in model_state.items())):
                _fail("model section is not a Tensor state mapping")
            if not isinstance(restored["metadata"], dict):
                _fail("metadata section is not a mapping")
        transaction = RestoreTransaction()
        transaction.prepare(model, optimizer, restored, target, absent)
        transaction.commit()
        return restored["metadata"]
