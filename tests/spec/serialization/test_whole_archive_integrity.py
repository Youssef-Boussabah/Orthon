"""Strict parsing and integrity of unrequested present sections."""
import json
import re
import warnings
import zipfile

import pytest
import orthon as ortho
from .workload import RegressionModel


@pytest.mark.parametrize('section', ['model', 'optimizer', 'metadata'])
@pytest.mark.parametrize('damage', ['checksum', 'missing-payload', 'compressed', 'unsupported-tag'])
def test_unrequested_present_section_is_fully_validated(
    saved_training_checkpoint, checkpoint_archive, tmp_path, section, damage,
):
    archive = checkpoint_archive(saved_training_checkpoint)
    record = archive.records(section)[0]
    name = record['payload']
    if damage == 'checksum':
        raw = archive.payloads[name]
        archive.payloads[name] = bytes([raw[0] ^ 1]) + raw[1:]
    elif damage == 'missing-payload':
        del archive.payloads[name]
    elif damage == 'unsupported-tag':
        record['k'] = 'arbitrary-class'
    path = archive.write(tmp_path / 'bad.ckpt', compressed=[name] if damage == 'compressed' else [])
    with pytest.raises(RuntimeError, match='checkpoint:'):
        ortho.load_checkpoint(path)


@pytest.mark.parametrize('api', ['checkpoint', 'state-only'])
def test_unrequested_optimizer_corruption_rejects_before_model_application(
    saved_training_checkpoint, checkpoint_archive, rejection_target,
    assert_destination_unchanged, tmp_path, api,
):
    model, optimizer = rejection_target
    unchanged = assert_destination_unchanged(model, optimizer)
    archive = checkpoint_archive(saved_training_checkpoint)
    name = archive.records('optimizer')[0]['payload']
    archive.payloads[name] = b'bad'
    path = archive.write(tmp_path / 'bad.ckpt')
    with pytest.raises(RuntimeError, match='checkpoint:'):
        if api == 'checkpoint':
            ortho.load_checkpoint(path, model=model)
        else:
            ortho.load_state_dict(model, path)
    unchanged()


def test_manifest_crc_failure_has_checkpoint_prefix_and_preserves_destination(
    saved_training_checkpoint, rejection_target, assert_destination_unchanged, tmp_path,
):
    model, optimizer = rejection_target
    unchanged = assert_destination_unchanged(model, optimizer)
    path = tmp_path / 'manifest-crc.ckpt'
    # Keep manifest bytes valid JSON, but corrupt its central-directory CRC.
    # The archive library detects the mismatch before JSON or recipient work.
    raw = bytearray(saved_training_checkpoint.read_bytes())
    with zipfile.ZipFile(saved_training_checkpoint) as archive:
        offset = archive.start_dir
    assert raw[offset:offset + 4] == b'PK\x01\x02'
    raw[offset + 16] ^= 1
    path.write_bytes(raw)
    with pytest.raises(RuntimeError, match='checkpoint: cannot read manifest.json'):
        ortho.load_checkpoint(path, model=model, optimizer=optimizer)
    unchanged()


@pytest.mark.parametrize('member', ['manifest.json', 'tensors/000000.bin', 'inert.txt'])
def test_duplicate_zip_member_names_reject_even_when_inert(
    saved_training_checkpoint, tmp_path, member,
):
    path = tmp_path / 'duplicate.ckpt'
    with zipfile.ZipFile(saved_training_checkpoint) as original, zipfile.ZipFile(path, 'w') as out:
        for info in original.infolist():
            out.writestr(info, original.read(info.filename))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            if member == 'inert.txt':
                out.writestr(member, b'first')
                out.writestr(member, b'second')
            else:
                out.writestr(member, original.read(member))
    with pytest.raises(RuntimeError, match='duplicate ZIP member'):
        ortho.load_checkpoint(path)


@pytest.mark.parametrize('field', ['format', 'format_version', 'sections', 'nbytes'])
def test_duplicate_json_fields_reject_at_any_depth(
    saved_training_checkpoint, checkpoint_archive, tmp_path, field,
):
    archive = checkpoint_archive(saved_training_checkpoint)
    text = json.dumps(archive.manifest)
    if field == 'sections':
        text = text.replace('"sections":', '"sections": {}, "sections":', 1)
    else:
        pattern = r'"'+field+r'": ("[^"]*"|[0-9]+)'
        text = re.sub(pattern, lambda m: m[0]+', '+m[0], text, count=1)
    with pytest.raises(RuntimeError, match='duplicate JSON object field'):
        ortho.load_checkpoint(archive.write(tmp_path / 'json.ckpt', manifest_bytes=text.encode()))


@pytest.mark.parametrize('section', ['model', 'optimizer', 'metadata'])
def test_duplicate_encoded_dictionary_keys_reject(
    saved_training_checkpoint, checkpoint_archive, tmp_path, section,
):
    archive = checkpoint_archive(saved_training_checkpoint)
    items = archive.manifest['sections'][section]['v']
    items.append(items[0])
    with pytest.raises(RuntimeError, match='duplicate encoded dictionary key'):
        ortho.load_checkpoint(archive.write(tmp_path / 'dict.ckpt'))


@pytest.mark.parametrize('constant', ['NaN', 'Infinity', '-Infinity'])
def test_bare_nonstandard_json_constants_reject_but_tagged_values_restore(
    saved_training_checkpoint, checkpoint_archive, tmp_path, constant,
):
    archive = checkpoint_archive(saved_training_checkpoint)
    text = json.dumps(archive.manifest)[:-1] + ', "unused": '+constant+'}'
    with pytest.raises(RuntimeError, match='non-standard JSON constant'):
        ortho.load_checkpoint(archive.write(tmp_path / 'constant.ckpt', manifest_bytes=text.encode()))
    path = tmp_path / 'tagged.ckpt'
    ortho.save_checkpoint(path, metadata={'value': float(constant.replace('Infinity', 'inf'))})
    value = ortho.load_checkpoint(path)['value']
    assert value != value if constant == 'NaN' else value == float(constant.replace('Infinity', 'inf'))


def test_payload_uniqueness_spans_unrequested_sections(
    saved_training_checkpoint, checkpoint_archive, tmp_path,
):
    archive = checkpoint_archive(saved_training_checkpoint)
    archive.records('optimizer')[0].update(archive.records('model')[0])
    with pytest.raises(RuntimeError, match='referenced more than once'):
        ortho.load_checkpoint(archive.write(tmp_path / 'shared.ckpt'))


@pytest.mark.parametrize('section,node', [
    ('model', {'k': 'list', 'v': []}), ('model', {'k': 'null'}),
    ('model', {'k': 'dict', 'v': [['wrong', {'k': 'int', 'v': 1}]]}),
    ('metadata', {'k': 'list', 'v': []}), ('metadata', {'k': 'null'}),
])
def test_present_section_schema_is_validated_without_recipient(
    saved_training_checkpoint, checkpoint_archive, tmp_path, section, node,
):
    archive = checkpoint_archive(saved_training_checkpoint)
    archive.manifest['sections'][section] = node
    with pytest.raises(RuntimeError, match='checkpoint:'):
        ortho.load_checkpoint(archive.write(tmp_path / 'schema.ckpt'))


def test_unique_extra_members_remain_inert_and_absent_sections_remain_valid(tmp_path):
    path = tmp_path / 'extra.ckpt'
    ortho.save_checkpoint(path, metadata={'x': 1})
    with zipfile.ZipFile(path, 'a') as archive:
        archive.writestr('../inert-data', b'never extracted')
    assert ortho.load_checkpoint(path) == {'x': 1}
    assert set(tmp_path.iterdir()) == {path}


def test_tensor_metadata_snapshots_remain_independent_cpu_leaves(operator_device, tmp_path):
    value = ortho.ones([2]).to(operator_device).requires_grad_(True)
    path = tmp_path / 'metadata.ckpt'
    ortho.save_checkpoint(path, metadata={'a': value, 'b': value})
    metadata = ortho.load_checkpoint(path, map_location=operator_device)
    for item in metadata.values():
        assert item.device.is_cpu() and item.is_leaf and not item.requires_grad and item.grad is None
    metadata['a'].mul_(2)
    assert metadata['b'].tolist() == [1, 1]
    assert value.tolist() == [1, 1]


def _r3_manifest_bytes():
    return json.dumps({
        "format": "ml-runtime-checkpoint", "format_version": 1,
        "sections": {"model": None, "optimizer": None,
                     "metadata": {"k": "dict", "v": []}},
    }, separators=(",", ":")).encode()


@pytest.mark.parametrize("extra", [0, 1])
def test_r3_manifest_byte_boundary_with_small_test_limit(monkeypatch, tmp_path, extra):
    from orthon._serialization import archive as codec
    blob = _r3_manifest_bytes()
    assert codec._MAX_MANIFEST_BYTES == 8 * 1024 * 1024
    monkeypatch.setattr(codec, "_MAX_MANIFEST_BYTES", len(blob))
    path = tmp_path / "boundary.ckpt"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", blob + b" " * extra)
    if extra:
        with pytest.raises(RuntimeError, match="checkpoint: manifest.json exceeds metadata limit"):
            ortho.load_checkpoint(path)
    else:
        assert ortho.load_checkpoint(path) == {}


def test_r3_declared_manifest_limit_rejects_before_member_open(monkeypatch):
    from orthon._serialization import archive as codec

    class DeclaredOversize:
        def namelist(self):
            return ["manifest.json"]

        def getinfo(self, name):
            info = zipfile.ZipInfo(name)
            info.file_size = codec._MAX_MANIFEST_BYTES + 1
            return info

        def open(self, *args):
            raise AssertionError("oversized declared metadata must not be opened")

    with pytest.raises(RuntimeError, match="checkpoint: manifest.json exceeds metadata limit"):
        codec.ArchiveCodec.read_manifest(DeclaredOversize())


def test_r3_manifest_stream_limit_independent_of_declared_size(monkeypatch):
    import io
    from orthon._serialization import archive as codec
    monkeypatch.setattr(codec, "_MAX_MANIFEST_BYTES", 256)
    calls = []

    class Member(io.BytesIO):
        def read(self, size=-1):
            calls.append(size)
            assert size == 257
            return super().read(size)

    class MisdeclaredSmall:
        def namelist(self):
            return ["manifest.json"]

        def getinfo(self, name):
            info = zipfile.ZipInfo(name)
            info.file_size = 1
            return info

        def open(self, info, mode):
            assert info.filename == "manifest.json" and mode == "r"
            return Member(b"x" * 512)

        def read(self, *args):
            raise AssertionError("unrestricted manifest read is forbidden")

    with pytest.raises(RuntimeError, match="checkpoint: manifest.json exceeds metadata limit"):
        codec.ArchiveCodec.read_manifest(MisdeclaredSmall())
    assert calls == [257]


def test_r3_compressible_manifest_is_bounded_before_recipient_mutation(
    monkeypatch, saved_training_checkpoint, checkpoint_archive,
    rejection_target, assert_destination_unchanged, tmp_path,
):
    from orthon._serialization import archive as codec
    model, optimizer = rejection_target
    unchanged = assert_destination_unchanged(model, optimizer)
    archive = checkpoint_archive(saved_training_checkpoint)
    archive.manifest["padding"] = "x" * 4096
    blob = json.dumps(archive.manifest).encode()
    monkeypatch.setattr(codec, "_MAX_MANIFEST_BYTES", len(blob) - 1)
    path = tmp_path / "compressed.ckpt"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as output:
        output.writestr("manifest.json", blob)
    assert path.stat().st_size < len(blob)
    with pytest.raises(RuntimeError, match="checkpoint: manifest.json exceeds metadata limit"):
        ortho.load_checkpoint(path, model=model, optimizer=optimizer)
    unchanged()


@pytest.mark.parametrize("blob,message", [
    (b"\xff", "not valid JSON"),
    (b"{", "not valid JSON"),
    (b'{"format":1,"format":2}', "duplicate JSON object field"),
    (b'{"unused":NaN}', "non-standard JSON constant"),
    (b'{"unused":Infinity}', "non-standard JSON constant"),
    (b'{"format":"ml-runtime-checkpoint","format_version":2}', "unsupported version"),
])
def test_r3_malformed_manifest_errors_are_managed(tmp_path, blob, message):
    path = tmp_path / "malformed.ckpt"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", blob)
    with pytest.raises(RuntimeError, match="checkpoint:.*" + message):
        ortho.load_checkpoint(path)


@pytest.mark.parametrize("extra", [0, 1])
def test_r3_json_nesting_boundary_is_independent_of_string_brackets(tmp_path, extra):
    from orthon._serialization import archive as codec
    # Root object counts as one level; an inert field can exercise exact depth.
    manifest = json.loads(_r3_manifest_bytes())
    prefix = json.dumps(manifest)[:-1] + ', "unused":'
    depth = codec._MAX_MANIFEST_DEPTH - 1 + extra
    blob = (prefix + "[" * depth + "0" + "]" * depth + "}").encode()
    path = tmp_path / "nested.ckpt"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", blob)
    if extra:
        with pytest.raises(RuntimeError, match="checkpoint:.*maximum JSON nesting"):
            ortho.load_checkpoint(path)
    else:
        assert ortho.load_checkpoint(path) == {}
    strings = tmp_path / "strings.ckpt"
    value = ('\\"[{}]\\\\' * 256)
    ortho.save_checkpoint(strings, metadata={"brackets": value})
    assert ortho.load_checkpoint(strings) == {"brackets": value}


def test_r3_writer_size_limit_rejects_before_temporary_or_replacement(monkeypatch, tmp_path):
    from orthon._serialization import archive as codec
    path = tmp_path / "prior.ckpt"
    ortho.save_checkpoint(path, metadata={"prior": True})
    before = path.read_bytes()
    with zipfile.ZipFile(path) as archive:
        limit = archive.getinfo("manifest.json").file_size
    monkeypatch.setattr(codec, "_MAX_MANIFEST_BYTES", limit)
    original = codec.tempfile.mkstemp
    calls = []

    def record_temporary(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(codec.tempfile, "mkstemp", record_temporary)
    with pytest.raises(RuntimeError, match="checkpoint:.*metadata limit"):
        ortho.save_checkpoint(path, metadata={"padding": "x" * 1024})
    assert path.read_bytes() == before and calls == []
    assert set(tmp_path.iterdir()) == {path}


def test_r3_writer_nesting_limit_preserves_existing_file(tmp_path):
    path = tmp_path / "prior.ckpt"
    ortho.save_checkpoint(path, metadata={"prior": True})
    before = path.read_bytes()
    value = 0
    for _ in range(70):
        value = [value]
    with pytest.raises(RuntimeError, match="checkpoint:.*maximum JSON nesting"):
        ortho.save_checkpoint(path, metadata={"nested": value})
    assert path.read_bytes() == before and set(tmp_path.iterdir()) == {path}


def test_r3_missing_or_truncated_manifest_remains_managed(tmp_path):
    path = tmp_path / "missing.ckpt"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("inert", b"x")
    with pytest.raises(RuntimeError, match="checkpoint: archive has no manifest.json"):
        ortho.load_checkpoint(path)
    ortho.save_checkpoint(path, metadata={"valid": True})
    path.write_bytes(path.read_bytes()[:-12])
    with pytest.raises(RuntimeError, match="checkpoint: corrupt archive"):
        ortho.load_checkpoint(path)


def test_r3_manifest_limit_does_not_cap_stored_tensor_payload(monkeypatch, tmp_path):
    from orthon._serialization import archive as codec
    monkeypatch.setattr(codec, "_MAX_MANIFEST_BYTES", 1024)
    model = ortho.nn.Module()
    model.weight = ortho.nn.Parameter(ortho.ones([512]).to("cpu"))
    path = tmp_path / "payload.ckpt"
    ortho.save_checkpoint(path, model=model)
    with zipfile.ZipFile(path) as archive:
        assert archive.getinfo("manifest.json").file_size < 1024
        info = archive.getinfo("tensors/000000.bin")
        assert info.file_size == 2048 and info.compress_type == zipfile.ZIP_STORED
    ortho.load_checkpoint(path, model=model)
    assert model.weight.tolist() == [1.] * 512


def test_r3_json_parser_recursion_error_uses_checkpoint_category(monkeypatch, tmp_path):
    from orthon._serialization import archive as codec
    path = tmp_path / "valid.ckpt"
    ortho.save_checkpoint(path, metadata={})

    def fail(*args, **kwargs):
        raise RecursionError("bounded parser sentinel")

    monkeypatch.setattr(codec.json, "loads", fail)
    with pytest.raises(RuntimeError, match="checkpoint:.*not valid JSON.*bounded parser sentinel"):
        ortho.load_checkpoint(path)
