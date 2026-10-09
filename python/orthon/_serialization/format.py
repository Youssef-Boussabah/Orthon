"""Frozen v1 wire constants and uniform integrity errors."""
import re

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
    """Raise RuntimeError with the uniform checkpoint integrity prefix."""
    raise RuntimeError("checkpoint: " + message)
