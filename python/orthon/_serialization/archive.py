"""The sole ZIP/JSON v1 archive reader and deterministic atomic writer."""
import json
import os
import tempfile
import zipfile

from .format import (FORMAT_ID, FORMAT_VERSION, _MANIFEST_NAME, _ZIP_DATE_TIME,
                     _ZIP_EXTERNAL_ATTR, _fail)

_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_MAX_MANIFEST_DEPTH = 128


def _validate_manifest_depth(blob):
    depth = 0
    in_string = False
    escaped = False
    for char in blob:
        if in_string:
            if escaped:
                escaped = False
            elif char == 92:
                escaped = True
            elif char == 34:
                in_string = False
        elif char == 34:
            in_string = True
        elif char in (123, 91):
            depth += 1
            if depth > _MAX_MANIFEST_DEPTH:
                _fail("manifest.json exceeds maximum JSON nesting depth of %d"
                      % _MAX_MANIFEST_DEPTH)
        elif char in (125, 93):
            depth -= 1


def _validate_manifest_size(size):
    if size > _MAX_MANIFEST_BYTES:
        _fail("manifest.json exceeds metadata limit of %d bytes"
              % _MAX_MANIFEST_BYTES)


def _strict_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON object field %r" % key)
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("non-standard JSON constant %s" % value)


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
    """Atomically replace the archive using a temporary in the destination directory.

    Same-filesystem rename preserves the existing file on failure; always
    remove the temporary."""
    target = os.path.abspath(path)
    parent = os.path.dirname(target) or os.curdir
    if not os.path.isdir(parent):
        _fail("destination directory does not exist: %s" % parent)

    # Non-finite floats use tagged strings; allow_nan=False rejects raw JSON NaNs
    # by the encoder, so if one ever reached json it would be a codec bug, and
    # this makes that fail loudly instead of writing NaN/Infinity literals that
    # are not valid JSON.
    try:
        blob = json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                          allow_nan=False, ensure_ascii=True).encode("utf-8")
    except RecursionError as exc:
        _fail("cannot encode manifest.json (%s)" % exc)
    _validate_manifest_size(len(blob))
    _validate_manifest_depth(blob)

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
    names = archive.namelist()
    if len(names) != len(set(names)):
        _fail("archive contains duplicate ZIP member names")
    try:
        info = archive.getinfo(_MANIFEST_NAME)
    except KeyError:
        _fail("archive has no %s" % _MANIFEST_NAME)
    _validate_manifest_size(info.file_size)
    try:
        with archive.open(info, "r") as member:
            blob = member.read(_MAX_MANIFEST_BYTES + 1)
    except Exception as exc:
        _fail("cannot read %s (%s)" % (_MANIFEST_NAME, exc))
    _validate_manifest_size(len(blob))
    _validate_manifest_depth(blob)
    try:
        manifest = json.loads(blob.decode("utf-8"), object_pairs_hook=_strict_object,
                              parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
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


class ArchiveCodec:
    fspath = staticmethod(_fspath)
    write = staticmethod(_write_archive)
    open = staticmethod(_open_archive)
    read_manifest = staticmethod(_read_manifest)

    @staticmethod
    def read_payload(archive, name, nbytes, path):
        # Nothing is ever extracted to disk: entries are read by name straight
        # out of the archive, so a crafted member name cannot write anywhere.
        try:
            info = archive.getinfo(name)
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
            payload = archive.read(name)
        except Exception as exc:
            _fail("%s: cannot read payload %r (%s)" % (path, name, exc))
        if len(payload) != nbytes:
            _fail("%s: payload %r read %d bytes, expected %d"
                  % (path, name, len(payload), nbytes))
        return payload
