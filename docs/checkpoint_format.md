# Checkpoint and State File Format

One versioned container serves all four file APIs — `save_state_dict`,
`load_state_dict`, `save_checkpoint`, `load_checkpoint` — implemented once in
`python/orthon/checkpoint.py`. There is no second serialization path to fall
out of step with it. How checkpointing sits in the runtime is sketched in
[architecture.md](architecture.md#checkpointing); how it interacts with
models, optimizers and AMP state is in
[training_stack.md](training_stack.md#checkpoint-interaction).

**There is no pickle anywhere in it.** Loading a checkpoint parses JSON and
copies bytes; it cannot execute code from the file.

## Why not pickle

A pickle container costs three things this format keeps:

| | pickle container | this format |
| --- | --- | --- |
| Tensor payload | `tensor.tolist()` | raw little-endian storage bytes |
| dtype fidelity | discarded, or re-derived through NumPy | recorded and copied |
| float64 round trip | lossy when dtype is dropped — 1/3 back as `0.3333333432674408` | exact |
| NaN payload | quieted (`0x7F800001` → `0x7FC00001`) | preserved |
| File size vs raw | ~2.3× | ~1× |
| Safe on untrusted input | **no** | yes |

The safety row is the decisive one. `pickle.load` runs a VM implementing
REDUCE / GLOBAL / STACK_GLOBAL, which import and call arbitrary Python named in
the stream, so loading an untrusted checkpoint is equivalent to executing it.

## Layout

```
manifest.json          UTF-8 JSON, deflated
tensors/000000.bin     raw little-endian tensor storage, STORED (uncompressed)
tensors/000001.bin
...
```

The manifest:

```json
{
  "format": "ml-runtime-checkpoint",
  "format_version": 1,
  "sections": {
    "model":     {"k": "dict", "v": [["0.bias", {"k": "tensor", ...}], ...]},
    "optimizer": {"k": "dict", "v": [...]} | null,
    "metadata":  {"k": "dict", "v": [...]}
  }
}
```

The format identifier is deliberately **brand-neutral**: it names the file
format, not the Python package, so the on-disk contract stands on its own and
is not coupled to any package's naming.

`format_version` starts at **1**. A legacy pickle payload also carried
`"version": 1`, but that is a different container entirely — the two numbers
are unrelated, and such files are rejected outright rather than
version-negotiated.

## Tensor records

```json
{
  "k": "tensor",
  "shape": [4, 2],
  "dtype": "float32",
  "nbytes": 32,
  "sha256": "…",
  "payload": "tensors/000000.bin"
}
```

The payload is exactly `numel * dtype_size(dtype)` bytes of logical contiguous
storage, produced by the private `_tensor_to_bytes` / `_tensor_from_bytes`
pair. No Python float list, no NumPy dtype table, no conversion.

**Device is not recorded.** Placement is a *load* policy, so the same
checkpoint loads onto CPU or CUDA depending on the destination or
`map_location`.

### Exact dtype and bit preservation

Every supported dtype round-trips bit-for-bit: `float32`, `float64`,
`float16`, `bfloat16`, `int32`, `int64`, `uint8`, `int8`, `bool`. So do signed
zero, infinities, and NaN *payloads* — including signalling NaNs, which the old
Tensor → Python float → NumPy pipeline quietly quieted.

Because restore copies bytes rather than re-converting, a stored float16 or
bfloat16 payload is **never re-rounded**. Checkpoint fidelity is therefore
independent of the host conversion policy in
`include/orthon/core/float_formats.h`.

A non-contiguous source is serialized in **logical row-major order**: saving
takes a `to("cpu")` snapshot under `no_grad`, which materializes views. A view
never contributes the unused parts of its backing storage.

## Safe value codec

Optimizer state and metadata are nested Python structures, encoded with a
closed, tagged whitelist:

| Python | Encoding |
| --- | --- |
| `None` | `{"k": "null"}` |
| `bool` | `{"k": "bool", "v": true}` |
| `int` | `{"k": "int", "v": 7}` |
| `float` (finite) | `{"k": "float", "v": 1.5}` |
| `float` (±inf, nan) | `{"k": "float", "v": "inf" \| "-inf" \| "nan"}` |
| `str` | `{"k": "str", "v": "…"}` |
| `list` | `{"k": "list", "v": [...]}` |
| `tuple` | `{"k": "tuple", "v": [...]}` |
| `dict` | `{"k": "dict", "v": [[key, value], ...]}` |
| `Tensor` | the tensor record above |

- The `list` / `tuple` distinction is preserved.
- `bool` is checked before `int`, since `bool` subclasses `int` in Python.
- Dict keys must be `str`. Every optimizer and metadata state in this runtime
  uses string keys; anything else raises `TypeError`.
- Dicts encode as an item **list**, not a JSON object, so a user key such as
  `"k"` can never be mistaken for a tag. There is no reserved-key collision.
- Non-finite floats are tagged as strings, so the manifest never needs JSON's
  non-standard `NaN` / `Infinity` literals; the writer passes `allow_nan=False`
  so a leak would fail loudly.
- Anything else — an arbitrary object, a class, `bytes`, `set`, `complex` — is
  a `TypeError` naming the path, e.g.
  `checkpoint: metadata.outer.inner[1]: unsupported value type <class 'object'>`.

Tensor **values** need none of this: they are raw bytes.

## Determinism

Saving the same logical state twice produces a byte-identical file. Nothing
environmental is recorded — no absolute paths, hostname, username, UUID or
wall-clock timestamp. ZIP entries use a fixed `date_time` of
`(1980, 1, 1, 0, 0, 0)` and a pinned `create_system`, tensor IDs follow a
deterministic traversal, JSON is written with sorted keys and fixed separators,
and model state names are sorted lexically before IDs are assigned.

Sorting happens **in the file only**. `Module.state_dict()` keeps its insertion
order.

## Atomic save

A checkpoint is written to a temporary file in the destination directory,
flushed, `fsync`ed and then `os.replace`d into place. If serialization fails
part-way — an unsupported metadata value, say — the existing checkpoint is left
byte-for-byte untouched and the temporary is removed. No `.tmp` files are left
behind. The parent directory must already exist; no directory trees are created
silently.

## Loading

`load_checkpoint` validates and decodes **everything** before touching the
destination:

1. identify the file by content (never by extension);
2. parse and validate the manifest — identifier, version, required sections;
3. for each tensor: validate the payload name, check the archive's stored size
   against the declared `nbytes`, read, verify SHA-256;
4. construct independent restored tensors;
5. decode optimizer state and metadata;
6. only then mutate the model and optimizer.

A malformed later tensor therefore cannot leave earlier parameters already
changed.

Rejected clearly, all with a `checkpoint:` prefix: a missing or malformed
manifest, an unknown format identifier, an unsupported version, a missing
section, a missing payload, a byte-count mismatch, a SHA-256 mismatch, a
payload referenced twice, a malformed dtype or shape, a shape inconsistent with
its payload, an unsupported encoded value kind, a corrupt ZIP, and a random
non-archive file.

### Legacy pickle files are not auto-loaded

A pickle file is detected by its header and refused **without unpickling**:

```
checkpoint: legacy pickle checkpoints are unsupported; this build reads only
ml-runtime-checkpoint v1 archives. Re-save the file with a trusted legacy
build before importing it
```

Detecting a pickle and then calling `pickle.load` would simply reinstate the
unsafe code path, so it is deliberately not offered.

### Path safety

Nothing is ever extracted to disk. Entries are read by name straight out of the
archive, and payload names are validated against an exact allowlist
(`tensors/NNNNNN.bin`) — strictly stronger than filtering for `..`, leading
slashes or backslashes. `extractall()` is never called.

### Resource sanity

Tensor payload entries must be `ZIP_STORED` (uncompressed) — exactly what the
v1 writer emits. A compressed payload entry is rejected *before* any read, so
a small archive can never decompress into a large allocation. A record's
`nbytes` must equal the archive's stored size for that entry, also checked
before the read, and `_tensor_from_bytes` independently re-derives the
expected length from shape × dtype size with overflow-checked arithmetic. A
manifest claiming an enormous tensor therefore fails on disagreement rather
than causing a large allocation. There is no arbitrary model-size cap.

## Device and `map_location`

`map_location` accepts a device string (`"cpu"`, `"cuda"`, `"cuda:1"`) or a
`Device`; it is normalized to a `Device` at the API boundary so every consumer
sees one type.

**Default (`map_location=None`)** — the destination is authoritative.
Model state loads into the model's *current* parameter and buffer devices, and
loading never moves a model unexpectedly. Optimizer tensor state is restored on
the CPU and placed by the optimizer's own `load_state_dict` on its parameter
devices.

**Explicit `map_location`** — the model is intentionally moved there before the
state copy, and optimizer tensor state targets it directly.

Model state is copied *into* existing parameters, never replacing them, so
optimizer references, post-accumulate hooks, `requires_grad` / `is_leaf` and
`.grad` all survive a load.

## Snapshot vs live reference

`Module.state_dict()` returns **live references** to the current state tensors:
mutating a parameter afterwards is visible through a `state_dict` taken
earlier. That is deliberate and unchanged.

The file APIs are the opposite by design: `save_state_dict` and
`save_checkpoint` take a **snapshot** of the values at save time, so mutating
the model afterwards cannot change what was written. Saving is otherwise
observational — it does not touch a tensor's storage version, `requires_grad`,
`.grad`, `grad_fn`, device or dtype.

## Dependencies

Standard library only: `zipfile`, `json`, `hashlib`, `tempfile`, `os`, `re`.
NumPy is a declared project dependency for other APIs, but checkpoint
loading does not need it to reconstruct a tensor.

## Inspecting a file

The format is designed to be readable without running any of this project's
code:

```python
import json, zipfile

with zipfile.ZipFile("model.ckpt") as archive:
    manifest = json.loads(archive.read("manifest.json"))
    for name, record in manifest["sections"]["model"]["v"]:
        print(name, record["dtype"], record["shape"], record["nbytes"])
    raw = archive.read("tensors/000000.bin")   # plain little-endian bytes
```

There is deliberately no CLI.
