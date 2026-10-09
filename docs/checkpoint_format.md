# Checkpoint and State File Format

One versioned container serves all four file APIs — `save_state_dict`,
`load_state_dict`, `save_checkpoint`, `load_checkpoint` — implemented once in
`python/orthon/checkpoint.py` as a thin facade over private services in
`python/orthon/_serialization/`. ArchiveCodec owns ZIP/JSON IO, ValueCodec owns
the closed value vocabulary, TensorSnapshotCodec owns native byte conversion,
and SerializationService coordinates a RestoreTransaction through each state
owner's private preparation protocol. There is no parallel serializer. How checkpointing sits in the runtime is sketched in
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
| Executes file-specified Python code | yes | no |

The code-execution row distinguishes the formats. `pickle.load` runs a VM implementing
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

Saving the same **ordered logical state** twice produces a byte-identical file. Nothing
environmental is recorded — no absolute paths, hostname, username, UUID or
wall-clock timestamp. ZIP entries use a fixed `date_time` of
`(1980, 1, 1, 0, 0, 0)` and a pinned `create_system`, tensor IDs follow a
deterministic traversal, JSON is written with sorted keys and fixed separators,
and model state names are sorted lexically before IDs are assigned.

Sorting happens **in the file only**. `Module.state_dict()` keeps its insertion
order. Encoded user dictionaries also retain insertion order in their item
lists; equal dictionaries with different insertion orders are not promised a
canonical archive. JSON object keys are sorted, which does not sort those lists.

## Atomic save

A checkpoint is written to a temporary file in the destination directory,
flushed, `fsync`ed and then `os.replace`d into place. If serialization fails
part-way — an unsupported metadata value, say — the existing checkpoint is left
byte-for-byte untouched and the temporary is removed. No `.tmp` files are left
behind. The parent directory must already exist; no directory trees are created
silently.

## Loading

`load_checkpoint` validates and decodes every present model, optimizer and
metadata section, including unrequested sections, before touching a recipient:

1. identify the file by content (never by extension) and reject duplicate ZIP
   member names, including duplicate inert members;
2. parse strict JSON, rejecting duplicate object fields at every depth and bare
   `NaN`, `Infinity` and `-Infinity`, then validate identifier/version/sections;
3. decode all sections in one session, rejecting duplicate encoded dictionary
   keys and repeated tensor payload references across sections;
4. validate stored byte counts, uncompressed payloads, SHA-256 and native
   shape/dtype byte arithmetic, constructing independent CPU tensor snapshots;
5. require present model state to be `dict[str, Tensor]` and metadata to be a
   dict; optimizer state remains a safe value tree whose concrete semantics
   belong to its recipient;
6. prepare the model and optimizer completely, including their state validation
   and target-device conversions, before either commits;
7. commit model first, then optimizer, and return metadata.

An absent model/optimizer section is valid unless that recipient was requested.
Unique unreferenced ZIP members remain inert and are never extracted or executed.
`load_state_dict` uses the same whole-file validation even for a combined file
whose optimizer state is not being applied.

Built-in Module, SGD, Adam and MixedPrecisionOptimizer share their private
prepare/commit paths with direct local state loading. Archive/schema, shape,
key, count, dtype/device conversion and optimizer semantic errors precede
publication. A synchronous commit exception attempts rollback in reverse order,
including the failing participant. Generic duck-typed recipients remain supported:
the adapter independently snapshots their public `state_dict()` and attempts
rollback through `load_state_dict()`. It does not inspect object dictionaries;
exact owner identity guarantees belong to built-in participants. Rollback failure
raises a checkpoint transaction RuntimeError chained from the original error.

This is a synchronous restore transaction, not recovery from arbitrary
asynchronous CUDA hardware failures after valid work is enqueued. It does not
add a device-wide synchronization to every load. Successful value writes and
rollback writes still follow normal Tensor mutation-generation rules.

Wire/format/integrity errors retain the `checkpoint:` RuntimeError prefix.
Recipient semantic errors retain their established exception categories.
SHA-256 detects corruption or payload mismatch; it does not authenticate the
archive's author.

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

The decompressed `manifest.json` is limited to **8 MiB (8,388,608 bytes)**.
The reader checks the declared size before opening it and independently reads
at most the limit plus one byte. JSON nesting is limited to 128 object/array
levels, ignoring brackets inside strings. Malformed or oversized metadata
raises a `checkpoint:` RuntimeError before recipient preparation. The writer
enforces the same limits before creating a temporary archive or replacing an
existing file. Ordinary valid archives retain the v1 format and deterministic bytes.

Tensor payload entries must be `ZIP_STORED` (uncompressed) — exactly what the
v1 writer emits. A compressed payload entry is rejected *before* any read.
These payload checks are distinct from the compressed manifest limit. A record's
`nbytes` must equal the archive's stored size for that entry, also checked
before the read, and `_tensor_from_bytes` independently re-derives the
expected length from shape × dtype size with overflow-checked arithmetic. A
manifest claiming an enormous tensor therefore fails on disagreement rather
than causing a large allocation. The manifest limit imposes no blanket maximum
total model payload size or separate tensor-count cap.

## Device and `map_location`

`map_location` accepts a device string (`"cpu"`, `"cuda"`, `"cuda:1"`) or a
`Device`; it is normalized to a `Device` at the API boundary so every consumer
sees one type.

**Default (`map_location=None`)** — the destination is authoritative.
Model state loads into the model's *current* parameter and buffer devices, and
loading never moves a model unexpectedly. Optimizer tensor state is restored on
the CPU and placed by the optimizer's own `load_state_dict` on its parameter
devices.

**Explicit `map_location`** — target placement is prepared before model
publication. Parameter Python owners survive a true device move; pre-existing
optimizers stay bound to those same owners. The optimizer receives the model's
intended future placement during preparation, so supported combined remapping
can restore moments/masters before the model commits. Optimizer-only placement
cannot move bound parameters: a conflicting explicit target rejects before
publication. SGD's direct configuration-only loader ignores `map_location`; checkpoint transactions validate explicit
placement using the same configuration preparation path.

Default model loading copies values into existing parameters, preserving
requires-grad/leaf metadata, hooks and existing gradients. Persistent buffers
receive independent prepared values with the destination requires-grad flag;
nonpersistent buffers are omitted from loaded state. A real explicit device move
uses the stable-owner representation transform and clears parameter gradients;
registered nonpersistent buffers move too. An explicit target already matching
an owner's device avoids unnecessary representation rebinding and preserves its
gradient. Existing destination parameter ties survive; file snapshots do not
reconstruct source alias graphs.

Metadata tensors always return as independent detached CPU leaves, regardless
of `map_location`. Metadata, training mode, gradients, RNG and scaler state are
not automatically applied. There are no new format sections.

## Snapshot vs live reference

`Module.state_dict()` returns **live references** to the current state tensors:
mutating a parameter afterwards is visible through a `state_dict` taken
earlier. That is deliberate and unchanged.

The file APIs are the opposite by design: `save_state_dict` and
`save_checkpoint` take a **snapshot** of the values at save time, so mutating
the model afterwards cannot change what was written. Saving is otherwise
observational — it does not touch a tensor's Buffer mutation generation, `requires_grad`,
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

## Service ownership and scope

Private implementations are `python/orthon/_serialization/format.py`,
`python/orthon/_serialization/archive.py`, `python/orthon/_serialization/codec.py`,
`python/orthon/_serialization/tensor.py`, `python/orthon/_serialization/service.py`
and `python/orthon/_serialization/transaction.py`. They are not public APIs.
The four root helpers keep the file signatures in [api.md](api.md).

This is a local single-file checkpoint. ProcessGroup, Reducer, rank/world-size
and shard coordination are not serialized as runtime services. Saving explicit
application metadata does not add a distributed checkpoint protocol.
