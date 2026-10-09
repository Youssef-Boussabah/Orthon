"""Graph-free logical tensor snapshots and the one native byte boundary."""
import hashlib

import orthon as ortho
from .format import _PAYLOAD_DIR, _PAYLOAD_RE, _fail


class TensorSnapshotCodec:
    @staticmethod
    def snapshot(tensor):
        """Copy logical values into an independent contiguous CPU tensor.

        Use to() under no_grad to materialize views and CUDA data without
        constructing a graph or mutating any source state."""
        with ortho.no_grad():
            return tensor.to("cpu")

    def encode(self, tensor, index):
        snapshot = self.snapshot(tensor)
        payload = ortho._tensor_to_bytes(snapshot)
        name = "%s/%06d.bin" % (_PAYLOAD_DIR, index)
        record = {
            "k": "tensor",
            "shape": [int(dim) for dim in snapshot.shape],
            "dtype": snapshot.dtype,
            "nbytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "payload": name,
        }

        return record, (name, payload)

    def decode(self, node, path, read_payload, seen_payloads):
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
        if not isinstance(name, str) or not _PAYLOAD_RE.fullmatch(name):
            _fail("%s: illegal tensor payload name %r" % (path, name))
        if name in seen_payloads:
            _fail("%s: payload %r is referenced more than once" % (path, name))
        seen_payloads.add(name)

        payload = read_payload(name, nbytes, path)
        if hashlib.sha256(payload).hexdigest() != digest:
            _fail("%s: tensor payload checksum mismatch for %r" % (path, name))

        # _tensor_from_bytes re-derives the expected length from shape/dtype
        # and rejects any mismatch, so a manifest whose nbytes agrees with the
        # stored blob but not with its own shape/dtype still fails here.
        try:
            tensor = ortho._tensor_from_bytes(shape, dtype, payload)
        except Exception as exc:
            _fail("%s: cannot construct tensor (%s)" % (path, exc))
        return tensor
