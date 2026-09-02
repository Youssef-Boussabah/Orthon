"""Versioned, pickle-free checkpoint and state file format.

One codec serves all four file APIs - ``save_state_dict``,
``load_state_dict``, ``save_checkpoint`` and ``load_checkpoint``.

WHY NOT PICKLE
--------------
The pickle VM implements REDUCE / GLOBAL / STACK_GLOBAL, which import and call
arbitrary Python named in the stream, so ``pickle.load`` on an untrusted
checkpoint is equivalent to executing it. Nothing here reads pickle, and a
legacy pickle file is refused by its header rather than opened.

Pickling a tensor also loses information that matters. Storing
``tensor.tolist()`` discards dtype, so a float64 parameter comes back rounded
to float32 (1/3 as 0.3333333432674408); a Tensor -> float -> NumPy pipeline
quiets a signalling NaN's payload (0x7F800001 to 0x7FC00001); and a Python
float object per element inflates the file about 2.3x. Raw bytes have none of
those problems.

THE FORMAT
----------
One ZIP container, one JSON manifest, one raw binary blob per tensor::

    manifest.json
    tensors/000000.bin
    tensors/000001.bin
    ...

* No pickle, no marshal, no eval/exec. Loading parses JSON and copies bytes;
  it can never execute code from the file.
* Tensor payloads are RAW little-endian storage bytes. No Python float list,
  no NumPy dtype table, no conversion - which is what makes float16 and
  bfloat16 survive bit-for-bit regardless of the host rounding policy, and
  what lets a NaN keep its payload.
* Every payload carries its byte count and a SHA-256, both checked before a
  tensor is constructed and long before the destination model is touched.
* Archives are deterministic: the same logical state produces byte-identical
  files. No timestamps, paths, hostnames or UUIDs are recorded.

The format identifier is deliberately BRAND-NEUTRAL: the on-disk format is
independent of what the Python package is called, so renaming the package
never requires a checkpoint migration.
"""
import hashlib
import json
import os
import re
import tempfile
import zipfile

import orthon as ortho

# ---------------------------------------------------------------------------
# Format constants
# ---------------------------------------------------------------------------

#: On-disk format identifier. Not "orthon": see the module docstring.
FORMAT_ID = "ml-runtime-checkpoint"

#: Starts at 1. A legacy pickle payload also carried ``"version": 1``, but
#: that is a different container entirely and the two numbers are unrelated -
#: such files are rejected outright, not version-negotiated.
FORMAT_VERSION = 1

_MANIFEST_NAME = "manifest.json"
_PAYLOAD_DIR = "tensors"
#: Payload names are validated against this exact pattern rather than being
#: filtered for "..", leading slashes or backslashes. An allowlist of the only
#: shape this writer ever produces is strictly stronger than any denylist, and
#: nothing is ever extracted to disk regardless (see _read_payload).
_PAYLOAD_RE = re.compile(r"^tensors/[0-9]{6}\.bin$")

#: Fixed ZIP metadata. A wall-clock timestamp would make two saves of the same
#: state differ, and create_system records the writing OS, so both are pinned.
_ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)
_ZIP_EXTERNAL_ATTR = 0o600 << 16

_NONFINITE_TO_TAG = {float("inf"): "inf", float("-inf"): "-inf"}
_TAG_TO_NONFINITE = {
    "inf": float("inf"),
    "-inf": float("-inf"),
    "nan": float("nan"),
}


def _fail(message):
    """Every file-format failure reads the same way.

    A plain RuntimeError rather than a private exception subclass: callers
    already catch RuntimeError from the previous API, and a leaked
    `_CheckpointError` name in a traceback would be noise. The uniform
    "checkpoint: " prefix is what makes these distinguishable.
    """
    raise RuntimeError("checkpoint: " + message)


# ---------------------------------------------------------------------------
# Tensor snapshot
# ---------------------------------------------------------------------------

def _snapshot(tensor):
    """Independent CPU, contiguous copy of a tensor's LOGICAL values.

    ``to()`` is the right primitive for all four things this needs at once: it
    is a physical copy, it materialises a strided view into row-major logical
    order, it crosses from CUDA, and it is exact-device aware. It is also
    differentiable, so the no_grad is explicit rather than incidental - saving
    a live requires-grad parameter must not build a transient ToNode.

    Saving is observational: nothing here mutates the source's values, storage
    version, requires_grad, grad, grad_fn, device or dtype.
    """
    with ortho.no_grad():
        return tensor.to("cpu")


# ---------------------------------------------------------------------------
# Safe recursive value codec
# ---------------------------------------------------------------------------
#
# Every encoded value is a tagged JSON object, so decoding is a closed
# whitelist dispatch with no ambiguity and no reserved-key collision: a user
# dict containing the key "k" cannot be mistaken for a tag, because dicts are
# encoded as an explicit item LIST rather than as a JSON object.
#
#   None    {"k": "null"}
#   bool    {"k": "bool",   "v": true}
#   int     {"k": "int",    "v": 7}
#   float   {"k": "float",  "v": 1.5}   finite
#           {"k": "float",  "v": "nan"} non-finite, tagged as a STRING so the
#                                       manifest never needs json's non-standard
#                                       NaN/Infinity literals (allow_nan=False)
#   str     {"k": "str",    "v": "..."}
#   list    {"k": "list",   "v": [...]}
#   tuple   {"k": "tuple",  "v": [...]}   distinct from list, by design
#   dict    {"k": "dict",   "v": [[key, value], ...]}
#   Tensor  {"k": "tensor", "shape": [...], "dtype": "...", "nbytes": N,
#            "sha256": "...", "payload": "tensors/000000.bin"}

_DICT_KEY_NOTE = (
    "dictionary keys must be strings; every optimizer and metadata state in "
    "this runtime uses string keys"
)


class _Encoder:
    """Turns a Python state tree into (manifest value, payload blobs)."""

    def __init__(self):
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
        snapshot = _snapshot(tensor)
        payload = ortho._tensor_to_bytes(snapshot)
        name = "%s/%06d.bin" % (_PAYLOAD_DIR, len(self.payloads))
        self.payloads.append((name, payload))
        return {
            "k": "tensor",
            "shape": [int(dim) for dim in snapshot.shape],
            "dtype": snapshot.dtype,
            "nbytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "payload": name,
        }


class _Decoder:
    """Turns a manifest value back into Python, reading payloads from `archive`.

    Constructed tensors are fresh CPU leaves unless a section asks for a
    device. One decoder handles every section of a load, so a payload
    referenced twice - across sections as well as within one - is caught.

    Entries the manifest does not reference are never read, so extra members in
    an archive are inert rather than an error; the safety property this format
    relies on is that nothing is ever extracted or executed, not that the
    container is byte-minimal.
    """

    def __init__(self, archive):
        self._archive = archive
        self._device = None
        self._seen_payloads = set()

    def decode_section(self, node, path, device=None):
        previous = self._device
        self._device = device
        try:
            return self.decode(node, path)
        finally:
            self._device = previous

    def decode(self, node, path):
        if not isinstance(node, dict):
            _fail("%s: expected an encoded object, got %s"
                  % (path, type(node).__name__))
        kind = node.get("k")
        handler = getattr(self, "_decode_" + str(kind), None)
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
            out[key] = self.decode(encoded, "%s.%s" % (path, key))
        return out

    def _decode_tensor(self, node, path):
        shape = node.get("shape")
        dtype = node.get("dtype")
        nbytes = node.get("nbytes")
        digest = node.get("sha256")
        name = node.get("payload")

        if not isinstance(shape, list) or not all(
                isinstance(d, int) and not isinstance(d, bool) for d in shape):
            _fail("%s: malformed tensor shape %r" % (path, shape))
        if any(d < 0 for d in shape):
            _fail("%s: negative dimension in shape %r" % (path, shape))
        if not isinstance(dtype, str):
            _fail("%s: malformed tensor dtype %r" % (path, dtype))
        if not isinstance(nbytes, int) or isinstance(nbytes, bool) or nbytes < 0:
            _fail("%s: malformed tensor nbytes %r" % (path, nbytes))
        if not isinstance(digest, str) or len(digest) != 64:
            _fail("%s: malformed tensor sha256" % path)
        if not isinstance(name, str) or not _PAYLOAD_RE.match(name):
            _fail("%s: illegal tensor payload name %r" % (path, name))
        if name in self._seen_payloads:
            _fail("%s: payload %r is referenced more than once" % (path, name))
        self._seen_payloads.add(name)

        payload = self._read_payload(name, nbytes, path)
        if hashlib.sha256(payload).hexdigest() != digest:
            _fail("%s: tensor payload checksum mismatch for %r" % (path, name))

        # _tensor_from_bytes re-derives the expected length from shape/dtype
        # and rejects any mismatch, so a manifest whose nbytes agrees with the
        # stored blob but not with its own shape/dtype still fails here.
        try:
            tensor = ortho._tensor_from_bytes(shape, dtype, payload)
        except Exception as exc:
            _fail("%s: cannot construct tensor (%s)" % (path, exc))
        if self._device is not None:
            with ortho.no_grad():
                tensor = tensor.to(self._device)
        return tensor

    def _read_payload(self, name, nbytes, path):
        # Nothing is ever extracted to disk: entries are read by name straight
        # out of the archive, so a crafted member name cannot write anywhere.
        try:
            info = self._archive.getinfo(name)
        except KeyError:
            _fail("%s: missing tensor payload %r" % (path, name))
        # The v1 writer stores tensor payloads uncompressed (ZIP_STORED), so
        # a compressed entry is by definition not a v1 checkpoint. Enforcing
        # that BEFORE the read also closes the decompression-amplification
        # hole: with STORED entries the on-disk size below is the true
        # in-memory size, so a tiny archive can never expand into a huge
        # allocation.
        if info.compress_type != zipfile.ZIP_STORED:
            _fail("%s: payload %r is compressed; v1 tensor payloads must be "
                  "stored uncompressed" % (path, name))
        # The STORED size is the authoritative bound, checked BEFORE the read,
        # so a manifest claiming an enormous tensor cannot cause a large
        # allocation - it simply disagrees with the archive and is rejected.
        if info.file_size != nbytes:
            _fail("%s: payload %r is %d bytes, manifest declares %d"
                  % (path, name, info.file_size, nbytes))
        try:
            payload = self._archive.read(name)
        except Exception as exc:
            _fail("%s: cannot read payload %r (%s)" % (path, name, exc))
        if len(payload) != nbytes:
            _fail("%s: payload %r read %d bytes, expected %d"
                  % (path, name, len(payload), nbytes))
        return payload


# ---------------------------------------------------------------------------
# Container IO
# ---------------------------------------------------------------------------

def _fspath(path, api):
    # bytes is deliberately refused even though os.fspath accepts it: it would
    # survive here and then fail deep inside os.path with "Can't mix bytes and
    # non-bytes in path components", which tells the caller nothing.
    if isinstance(path, bytes):
        raise TypeError(
            "%s: path must be a str or os.PathLike, got bytes" % api)
    try:
        resolved = os.fspath(path)
    except TypeError:
        raise TypeError(
            "%s: path must be a str or os.PathLike, got %r" % (api, type(path)))
    if not isinstance(resolved, str):
        raise TypeError(
            "%s: path must be a str or os.PathLike, got %r" % (api, type(path)))
    return resolved


def _write_archive(path, manifest, payloads):
    """Atomically write the archive.

    The temporary lives in the DESTINATION directory so os.replace is a same
    filesystem rename, which is atomic. Any failure leaves the existing file at
    `path` byte-for-byte untouched and removes the temporary.
    """
    target = os.path.abspath(path)
    parent = os.path.dirname(target) or os.curdir
    if not os.path.isdir(parent):
        _fail("destination directory does not exist: %s" % parent)

    # allow_nan=False is load-bearing: non-finite floats are tagged as strings
    # by the encoder, so if one ever reached json it would be a codec bug, and
    # this makes that fail loudly instead of writing NaN/Infinity literals that
    # are not valid JSON.
    blob = json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, ensure_ascii=True).encode("utf-8")

    handle, tmp = tempfile.mkstemp(prefix=".ckpt-", suffix=".tmp", dir=parent)
    try:
        with os.fdopen(handle, "wb") as raw:
            with zipfile.ZipFile(raw, "w") as archive:
                _write_entry(archive, _MANIFEST_NAME, blob,
                             zipfile.ZIP_DEFLATED)
                for name, payload in payloads:
                    # Tensor payloads are STORED, not deflated: raw float data
                    # barely compresses, and storing it keeps the archive's
                    # uncompressed size trivially authoritative on load.
                    _write_entry(archive, name, payload, zipfile.ZIP_STORED)
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _write_entry(archive, name, data, compress_type):
    info = zipfile.ZipInfo(name, date_time=_ZIP_DATE_TIME)
    info.compress_type = compress_type
    info.external_attr = _ZIP_EXTERNAL_ATTR
    info.create_system = 0        # pinned: otherwise it records the writing OS
    archive.writestr(info, data)


def _open_archive(path, api):
    """Open and identify the file WITHOUT ever unpickling it."""
    target = _fspath(path, api)
    try:
        with open(target, "rb") as handle:
            magic = handle.read(4)
    except OSError as exc:
        _fail("cannot open %r (%s)" % (target, exc))

    # Detection is by CONTENT, never by filename extension.
    if magic[:1] == b"\x80":
        # The PROTO opcode: a pickle written with protocol 2 or newer.
        # Rejected WITHOUT calling pickle.load - detecting a legacy file and
        # then unpickling it would simply reinstate the unsafe code path.
        _fail(
            "legacy pickle checkpoints are unsupported; this build reads only "
            "%s v%d archives. Re-save the file with a trusted legacy build "
            "before importing it"
            % (FORMAT_ID, FORMAT_VERSION))
    if magic[:2] != b"PK":
        _fail("not a %s archive (unrecognised file header)" % FORMAT_ID)

    try:
        return zipfile.ZipFile(target, "r")
    except zipfile.BadZipFile as exc:
        _fail("corrupt archive (%s)" % exc)


def _read_manifest(archive):
    try:
        blob = archive.read(_MANIFEST_NAME)
    except KeyError:
        _fail("archive has no %s" % _MANIFEST_NAME)
    try:
        manifest = json.loads(blob.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        _fail("%s is not valid JSON (%s)" % (_MANIFEST_NAME, exc))
    if not isinstance(manifest, dict):
        _fail("%s must be a JSON object" % _MANIFEST_NAME)

    identifier = manifest.get("format")
    if identifier != FORMAT_ID:
        _fail("unsupported format %r (expected %r)" % (identifier, FORMAT_ID))
    version = manifest.get("format_version")
    if not isinstance(version, int) or isinstance(version, bool):
        _fail("missing or malformed format_version")
    if version != FORMAT_VERSION:
        _fail("unsupported version %d (this build reads version %d)"
              % (version, FORMAT_VERSION))
    sections = manifest.get("sections")
    if not isinstance(sections, dict):
        _fail("manifest is missing its 'sections' object")
    for required in ("model", "optimizer", "metadata"):
        if required not in sections:
            _fail("manifest section %r is missing" % required)
    return sections


# ---------------------------------------------------------------------------
# Model state
# ---------------------------------------------------------------------------

def _normalize_map_location(spec, api):
    """Turn a device spec into a Device, once, at the API boundary.

    map_location is documented to accept either a device string or a Device.
    Downstream consumers - Adam._restore_state_tensors and
    MixedPrecisionOptimizer.load_state_dict - compare it against a tensor's
    .device, where a raw string would compare unequal to every Device.
    Normalising here means every consumer sees one type.
    """
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
    """Encode model.state_dict() with LEXICALLY SORTED names.

    state_dict() iteration follows module insertion order. Sorting here (and
    only here) makes the archive deterministic and its manifest diffable,
    without changing the in-memory ordering any caller sees.
    """
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


def _apply_model_state(model, restored, map_location):
    """Copy fully validated state into the destination.

    Module.load_state_dict copies INTO existing parameters rather than
    replacing them, so optimizer references, hooks, requires_grad/is_leaf and
    .grad all survive. Nothing here is reached until every payload has been
    checksummed and every tensor constructed.
    """
    if map_location is not None and hasattr(model, "to"):
        model.to(map_location)
    current = model.state_dict() if hasattr(model, "state_dict") else {}
    placed = {}
    for name, tensor in restored.items():
        # map_location has already been normalised to a Device, so this is a
        # Device-to-Device comparison and a same-device restore costs nothing.
        device = map_location
        if device is None and name in current:
            device = current[name].device
        if device is not None and tensor.device != device:
            with ortho.no_grad():
                tensor = tensor.to(device)
        placed[name] = tensor
    model.load_state_dict(placed)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_checkpoint(path, model=None, optimizer=None, metadata=None):
    """Write a checkpoint archive containing any of model / optimizer / metadata.

    The file is written atomically: an existing checkpoint at `path` is
    replaced only after the new archive is complete on disk.
    """
    target = _fspath(path, "save_checkpoint")
    encoder = _Encoder()

    model_section = None
    if model is not None:
        model_section = _encode_model(encoder, model, "save_checkpoint")

    optimizer_section = None
    if optimizer is not None:
        _require_state_dict(optimizer, "save_checkpoint", "optimizer")
        optimizer_section = encoder.encode(optimizer.state_dict(), "optimizer")

    metadata_section = encoder.encode(dict(metadata or {}), "metadata")

    manifest = {
        "format": FORMAT_ID,
        "format_version": FORMAT_VERSION,
        "sections": {
            "model": model_section,
            "optimizer": optimizer_section,
            "metadata": metadata_section,
        },
    }
    _write_archive(target, manifest, encoder.payloads)


def load_checkpoint(path, model=None, optimizer=None, map_location=None):
    """Restore model and/or optimizer state from an archive; return metadata.

    Everything is parsed, checksummed and decoded BEFORE the destination is
    touched, so a corrupt archive cannot leave a partially updated
    model behind.
    """
    map_location = _normalize_map_location(map_location, "load_checkpoint")
    with _open_archive(path, "load_checkpoint") as archive:
        sections = _read_manifest(archive)

        if model is not None and sections["model"] is None:
            _fail("archive has no model state")
        if optimizer is not None and sections["optimizer"] is None:
            _fail("archive has no optimizer state")
        if optimizer is not None and not hasattr(optimizer, "load_state_dict"):
            raise TypeError(
                "load_checkpoint: optimizer does not support load_state_dict()")

        # ---- decode everything first; nothing is mutated in this block ----
        decoder = _Decoder(archive)
        restored_model = None
        if model is not None:
            if not hasattr(model, "load_state_dict"):
                raise TypeError(
                    "load_checkpoint: model does not support load_state_dict()")
            # Model tensors are decoded on the CPU and placed afterwards, so
            # the destination-device decision stays in one place.
            decoded = decoder.decode_section(sections["model"], "model")
            if not isinstance(decoded, dict):
                _fail("model section is not a state mapping")
            restored_model = decoded

        optimizer_state = None
        if optimizer is not None:
            # Optimizer tensor state targets map_location directly when one is
            # given; otherwise it is decoded on the CPU and the optimizer's own
            # load_state_dict places it on the parameter devices.
            optimizer_state = decoder.decode_section(
                sections["optimizer"], "optimizer", device=map_location)

        metadata = decoder.decode_section(sections["metadata"], "metadata")

    # ---- commit ----
    if restored_model is not None:
        _apply_model_state(model, restored_model, map_location)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state, map_location=map_location)
    return metadata


def save_state_dict(model, path):
    """Write model.state_dict() to a state-only archive (no optimizer section).

    This is a SNAPSHOT: the file holds the values as of this call, and later
    mutation of the model cannot change it. Contrast with Module.state_dict(),
    which returns live references to the current state tensors.
    """
    save_checkpoint(path, model=model)


def load_state_dict(model, path, map_location=None):
    """Load a state archive into `model`.

    By default the model's CURRENT parameter and buffer devices are
    authoritative: loading never moves a model unexpectedly. Passing
    `map_location` intentionally moves the model there first, matching
    load_checkpoint's explicit-map contract.
    """
    load_checkpoint(path, model=model, map_location=map_location)
