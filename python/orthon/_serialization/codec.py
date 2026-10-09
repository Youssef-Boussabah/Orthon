"""Closed safe value vocabulary; one decoder session spans every section."""
import orthon as ortho
from .format import _NONFINITE_TO_TAG, _TAG_TO_NONFINITE, _fail
from .tensor import TensorSnapshotCodec


_DICT_KEY_NOTE = (
    "dictionary keys must be strings; every optimizer and metadata state in "
    "this runtime uses string keys"
)


class ValueEncoder:
    """Turns a Python state tree into (manifest value, payload blobs)."""

    def __init__(self):
        self._snapshots = TensorSnapshotCodec()
        self.payloads = []          # list of (archive_name, bytes)

    def encode(self, value, path):
        # bool BEFORE int: bool is a subclass of int in Python.
        if value is None:
            return {"k": "null"}
        if isinstance(value, bool):
            return {"k": "bool", "v": value}
        if isinstance(value, int):
            return {"k": "int", "v": value}
        if isinstance(value, float):
            if value != value:                      # NaN
                return {"k": "float", "v": "nan"}
            if value in _NONFINITE_TO_TAG:
                return {"k": "float", "v": _NONFINITE_TO_TAG[value]}
            return {"k": "float", "v": value}
        if isinstance(value, str):
            return {"k": "str", "v": value}
        if isinstance(value, ortho.Tensor):
            return self._encode_tensor(value)
        if isinstance(value, tuple):
            return {"k": "tuple",
                    "v": [self.encode(item, "%s[%d]" % (path, i))
                          for i, item in enumerate(value)]}
        if isinstance(value, list):
            return {"k": "list",
                    "v": [self.encode(item, "%s[%d]" % (path, i))
                          for i, item in enumerate(value)]}
        if isinstance(value, dict):
            items = []
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError(
                        "checkpoint: %s: unsupported dict key type %r (%s)"
                        % (path, type(key), _DICT_KEY_NOTE))
                items.append([key, self.encode(item, "%s.%s" % (path, key))])
            return {"k": "dict", "v": items}
        raise TypeError(
            "checkpoint: %s: unsupported value type %r" % (path, type(value)))

    def _encode_tensor(self, tensor):
        record, payload = self._snapshots.encode(tensor, len(self.payloads))
        self.payloads.append(payload)
        return record


class ValueDecoder:
    """Decode safe values into fresh CPU leaf tensors.

    One session rejects duplicate payload references across all sections.
    Unreferenced members remain inert: nothing is extracted or executed."""

    def __init__(self, read_payload):
        self._read_payload = read_payload
        self._snapshots = TensorSnapshotCodec()
        self._seen_payloads = set()

    def decode(self, node, path):
        if not isinstance(node, dict):
            _fail("%s: expected an encoded object, got %s"
                  % (path, type(node).__name__))
        kind = node.get("k")
        handlers = {name: getattr(self, "_decode_" + name) for name in
                    ("null", "bool", "int", "float", "str", "list", "tuple", "dict", "tensor")}
        handler = handlers.get(kind) if isinstance(kind, str) else None
        if handler is None:
            _fail("%s: unsupported encoded value kind %r" % (path, kind))
        return handler(node, path)

    @staticmethod
    def _value(node, path):
        if "v" not in node:
            _fail("%s: encoded value is missing its 'v' field" % path)
        return node["v"]

    def _decode_null(self, node, path):
        return None

    def _decode_bool(self, node, path):
        value = self._value(node, path)
        if not isinstance(value, bool):
            _fail("%s: expected a bool" % path)
        return value

    def _decode_int(self, node, path):
        value = self._value(node, path)
        if not isinstance(value, int) or isinstance(value, bool):
            _fail("%s: expected an int" % path)
        return value

    def _decode_float(self, node, path):
        value = self._value(node, path)
        if isinstance(value, str):
            if value not in _TAG_TO_NONFINITE:
                _fail("%s: unknown non-finite float tag %r" % (path, value))
            return _TAG_TO_NONFINITE[value]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _fail("%s: expected a float" % path)
        return float(value)

    def _decode_str(self, node, path):
        value = self._value(node, path)
        if not isinstance(value, str):
            _fail("%s: expected a str" % path)
        return value

    def _decode_list(self, node, path):
        return [self.decode(item, "%s[%d]" % (path, i))
                for i, item in enumerate(self._sequence(node, path))]

    def _decode_tuple(self, node, path):
        return tuple(self.decode(item, "%s[%d]" % (path, i))
                     for i, item in enumerate(self._sequence(node, path)))

    def _sequence(self, node, path):
        value = self._value(node, path)
        if not isinstance(value, list):
            _fail("%s: expected a sequence" % path)
        return value

    def _decode_dict(self, node, path):
        items = self._value(node, path)
        if not isinstance(items, list):
            _fail("%s: expected a dict item list" % path)
        out = {}
        for entry in items:
            if not isinstance(entry, list) or len(entry) != 2:
                _fail("%s: malformed dict entry" % path)
            key, encoded = entry
            if not isinstance(key, str):
                _fail("%s: dict key must be a string" % path)
            if key in out:
                _fail("%s: duplicate encoded dictionary key %r" % (path, key))
            out[key] = self.decode(encoded, "%s.%s" % (path, key))
        return out

    def _decode_tensor(self, node, path):
        return self._snapshots.decode(node, path, self._read_payload, self._seen_payloads)


class ValueCodec:
    """Factories for fresh value sessions; payload references span the session."""
    encoder = ValueEncoder
    decoder = ValueDecoder
