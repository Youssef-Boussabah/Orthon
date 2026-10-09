import hashlib
import io
import pickle

import pytest

import orthon as ortho
from orthon.checkpoint import FORMAT_ID, FORMAT_VERSION

from .workload import RegressionModel, StateSource


def test_documented_versioned_format_is_inspectable_through_standard_archive_tools(
    saved_training_checkpoint, checkpoint_archive,
):
    archive = checkpoint_archive(saved_training_checkpoint)
    manifest = archive.manifest
    assert manifest["format"] == FORMAT_ID == "ml-runtime-checkpoint"
    assert manifest["format_version"] == FORMAT_VERSION == 1
    assert {"model", "optimizer", "metadata"} <= manifest["sections"].keys()
    for section in ("model", "optimizer", "metadata"):
        for record in archive.records(section):
            payload = archive.payloads[record["payload"]]
            assert len(payload) == record["nbytes"]
            assert hashlib.sha256(payload).hexdigest() == record["sha256"]
            assert "device" not in record


@pytest.mark.parametrize("version", [0, FORMAT_VERSION + 1, True, None, "1"])
def test_unknown_or_malformed_versions_are_rejected(
    saved_training_checkpoint, checkpoint_archive, tmp_path, version,
):
    archive = checkpoint_archive(saved_training_checkpoint)
    archive.manifest["format_version"] = version
    path = archive.write(tmp_path / "version.ckpt")
    with pytest.raises(RuntimeError):
        ortho.load_checkpoint(path)


@pytest.mark.parametrize("field", ["format", "format_version", "sections"])
def test_required_manifest_fields_cannot_be_omitted(
    saved_training_checkpoint, checkpoint_archive, tmp_path, field,
):
    archive = checkpoint_archive(saved_training_checkpoint)
    del archive.manifest[field]
    with pytest.raises(RuntimeError):
        ortho.load_checkpoint(archive.write(tmp_path / "missing-field.ckpt"))


@pytest.mark.parametrize("section", ["model", "optimizer", "metadata"])
def test_required_section_slots_cannot_be_omitted(
    saved_training_checkpoint, checkpoint_archive, tmp_path, section,
):
    archive = checkpoint_archive(saved_training_checkpoint)
    del archive.manifest["sections"][section]
    with pytest.raises(RuntimeError):
        ortho.load_checkpoint(archive.write(tmp_path / "missing-section.ckpt"))


def test_unrecognized_format_identifier_is_rejected(saved_training_checkpoint, checkpoint_archive, tmp_path):
    archive = checkpoint_archive(saved_training_checkpoint)
    archive.manifest["format"] = "another-checkpoint-format"
    with pytest.raises(RuntimeError):
        ortho.load_checkpoint(archive.write(tmp_path / "format.ckpt"))


@pytest.mark.parametrize("blob", [b"not JSON", b"\xff", b"[]", b"null"])
def test_invalid_manifest_documents_are_rejected(
    saved_training_checkpoint, checkpoint_archive, tmp_path, blob,
):
    archive = checkpoint_archive(saved_training_checkpoint)
    with pytest.raises(RuntimeError):
        ortho.load_checkpoint(archive.write(tmp_path / "manifest.ckpt", manifest_bytes=blob))


def test_archive_without_manifest_is_rejected(saved_training_checkpoint, checkpoint_archive, tmp_path):
    archive = checkpoint_archive(saved_training_checkpoint)
    with pytest.raises(RuntimeError):
        ortho.load_checkpoint(archive.write(tmp_path / "missing.ckpt", omit_manifest=True))


@pytest.mark.parametrize("field", ["shape", "dtype", "nbytes", "sha256", "payload"])
def test_tensor_records_require_the_documented_reconstruction_and_integrity_fields(
    saved_training_checkpoint, checkpoint_archive, tmp_path, field,
):
    archive = checkpoint_archive(saved_training_checkpoint)
    del archive.records("metadata")[0][field]
    with pytest.raises(RuntimeError):
        ortho.load_checkpoint(archive.write(tmp_path / "tensor-field.ckpt"))


@pytest.mark.parametrize("path_form", ["path", "string"])
def test_file_identity_depends_on_content_and_accepts_public_path_forms(tmp_path, path_form):
    path = tmp_path / "valid.pkl"
    supplied = path if path_form == "path" else str(path)
    assert ortho.save_checkpoint(supplied, metadata={"step": 2}) is None
    assert ortho.load_checkpoint(supplied) == {"step": 2}


@pytest.mark.parametrize("api", ["save_checkpoint", "load_checkpoint", "save_state_dict", "load_state_dict"])
@pytest.mark.parametrize("bad_path", [None, 5, b"bytes-path", io.BytesIO()])
def test_file_apis_reject_non_path_inputs(api, bad_path):
    function = getattr(ortho, api)
    with pytest.raises(TypeError):
        if api.endswith("state_dict"):
            function(RegressionModel(), bad_path)
        else:
            function(bad_path)


@pytest.mark.parametrize("value", [object(), b"bytes", {1, 2}, 1 + 2j, lambda: None])
def test_unsupported_nested_state_cannot_be_serialized(tmp_path, value):
    with pytest.raises(TypeError):
        ortho.save_checkpoint(tmp_path / "unsupported.ckpt", metadata={"history": [0, value]})


def test_non_string_dictionary_keys_are_rejected_even_when_nested(tmp_path):
    with pytest.raises(TypeError):
        ortho.save_checkpoint(tmp_path / "keys.ckpt", metadata={"nested": [{1: "one"}]})


@pytest.mark.parametrize("role", ["model", "optimizer"])
def test_state_provider_is_required_for_save(tmp_path, role):
    with pytest.raises(TypeError):
        ortho.save_checkpoint(tmp_path / "provider.ckpt", **{role: object()})


def test_model_state_entries_must_be_tensors(tmp_path):
    with pytest.raises(TypeError):
        ortho.save_state_dict(StateSource({"weight": [1., 2.]}), tmp_path / "model.ckpt")


@pytest.mark.parametrize("role", ["model", "optimizer"])
def test_state_consumer_is_required_for_load(saved_training_checkpoint, role):
    with pytest.raises(TypeError):
        ortho.load_checkpoint(saved_training_checkpoint, **{role: object()})


@pytest.mark.parametrize("bad_location,error", [(3, TypeError), ([], TypeError), ("invalid-device", ValueError)])
def test_invalid_map_location_is_rejected_at_the_public_boundary(saved_training_checkpoint, bad_location, error):
    with pytest.raises(error):
        ortho.load_checkpoint(saved_training_checkpoint, map_location=bad_location)


def test_arbitrary_object_serialization_hooks_are_not_called(tmp_path):
    calls = []

    class Unsupported:
        def __reduce__(self):
            calls.append("reduce")
            return str, ("benign value",)

    with pytest.raises(TypeError):
        ortho.save_checkpoint(tmp_path / "hooks.ckpt", metadata={"value": Unsupported()})
    assert calls == []


@pytest.mark.parametrize("protocol", [0, pickle.HIGHEST_PROTOCOL])
def test_legacy_plain_data_is_refused_without_invoking_pickle_readers(
    tmp_path, monkeypatch, protocol,
):
    path = tmp_path / "legacy.data"
    path.write_bytes(pickle.dumps({"step": 1, "values": [2, 3]}, protocol=protocol))
    calls = []

    def record_read(*args, **kwargs):
        calls.append("pickle reader")
        raise AssertionError("checkpoint loading invoked an executable-object reader")

    monkeypatch.setattr(pickle, "load", record_read)
    monkeypatch.setattr(pickle, "loads", record_read)
    with pytest.raises(RuntimeError):
        ortho.load_checkpoint(path)
    assert calls == []


def test_unreferenced_archive_members_are_inert_data(
    saved_training_checkpoint, checkpoint_archive, tmp_path,
):
    archive = checkpoint_archive(saved_training_checkpoint)
    archive.payloads["unreferenced.txt"] = b"plain unused text"
    path = archive.write(tmp_path / "extra.ckpt")
    before = set(tmp_path.iterdir())
    result = ortho.load_checkpoint(path)
    assert result["step"] == 1
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize("name", ["../outside.txt", "/absolute.txt", "tensors/../outside.txt"])
def test_invalid_payload_names_are_rejected_without_filesystem_extraction(
    saved_training_checkpoint, checkpoint_archive, tmp_path, name,
):
    archive = checkpoint_archive(saved_training_checkpoint)
    archive.records("metadata")[0]["payload"] = name
    path = archive.write(tmp_path / "payload-name.ckpt")
    before = set(tmp_path.iterdir())
    with pytest.raises(RuntimeError):
        ortho.load_checkpoint(path)
    assert set(tmp_path.iterdir()) == before
