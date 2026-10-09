import pytest

import orthon as ortho

from .workload import RegressionModel, StateSource, advance, make_optimizer


@pytest.mark.parametrize("section", ["model", "optimizer", "metadata"])
@pytest.mark.parametrize("damage", [
    "checksum", "missing-payload", "short-payload", "shape", "dtype",
    "byte-count", "digest", "compressed",
])
def test_corruption_in_any_requested_section_preserves_entire_destination(
    saved_training_checkpoint, checkpoint_archive, rejection_target,
    assert_destination_unchanged, tmp_path, section, damage,
):
    model, optimizer = rejection_target
    unchanged = assert_destination_unchanged(model, optimizer)
    archive = checkpoint_archive(saved_training_checkpoint)
    record = archive.records(section)[-1]
    name = record["payload"]
    if damage == "checksum":
        payload = archive.payloads[name]
        archive.payloads[name] = bytes([payload[0] ^ 1]) + payload[1:]
    elif damage == "missing-payload":
        del archive.payloads[name]
    elif damage == "short-payload":
        archive.payloads[name] = archive.payloads[name][:-1]
    elif damage == "shape":
        record["shape"] = [99]
    elif damage == "dtype":
        record["dtype"] = "unsupported-dtype"
    elif damage == "byte-count":
        record["nbytes"] += 1
    elif damage == "digest":
        record["sha256"] = "short"
    path = archive.write(tmp_path / "corrupt.ckpt", compressed=[name] if damage == "compressed" else [])
    with pytest.raises(RuntimeError):
        ortho.load_checkpoint(path, model=model, optimizer=optimizer)
    unchanged()


@pytest.mark.parametrize("damage", ["metadata-kind", "metadata-scalar", "duplicate-reference"])
def test_malformed_decoded_state_is_rejected_before_any_commit(
    saved_training_checkpoint, checkpoint_archive, rejection_target,
    assert_destination_unchanged, tmp_path, damage,
):
    model, optimizer = rejection_target
    unchanged = assert_destination_unchanged(model, optimizer)
    archive = checkpoint_archive(saved_training_checkpoint)
    if damage == "metadata-kind":
        archive.manifest["sections"]["metadata"] = {"k": "python-object", "v": {"description": "plain data"}}
    elif damage == "metadata-scalar":
        archive.manifest["sections"]["metadata"] = {"k": "bool", "v": 1}
    else:
        first = archive.records("model")[0]
        archive.records("metadata")[0].update(first)
    with pytest.raises(RuntimeError):
        ortho.load_checkpoint(archive.write(tmp_path / "decoded.ckpt"), model=model, optimizer=optimizer)
    unchanged()


@pytest.mark.parametrize("damage,error", [("shape", ValueError), ("missing", KeyError), ("unexpected", KeyError)])
def test_model_mapping_validation_preserves_model_and_optimizer_before_commit(
    rejection_target, assert_destination_unchanged, tmp_path, damage, error,
):
    model, optimizer = rejection_target
    unchanged = assert_destination_unchanged(model, optimizer)
    source = RegressionModel()
    source.weight.mul_(3.)
    state = dict(source.state_dict())
    if damage == "shape":
        state["head.bias"] = ortho.full([2], 4.)
    elif damage == "missing":
        del state["head.bias"]
    else:
        state["extra"] = ortho.full([1], 4.)
    path = tmp_path / "mapping.ckpt"
    ortho.save_checkpoint(path, model=StateSource(state), optimizer=make_optimizer(source, "Adam"))
    with pytest.raises(error):
        ortho.load_checkpoint(path, model=model, optimizer=optimizer)
    unchanged()


@pytest.mark.parametrize("present", ["neither", "model", "optimizer"])
def test_absent_requested_state_is_rejected_before_changing_destination(
    rejection_target, assert_destination_unchanged, tmp_path, present,
):
    model, optimizer = rejection_target
    unchanged = assert_destination_unchanged(model, optimizer)
    source = RegressionModel()
    options = {"model": source} if present == "model" else (
        {"optimizer": make_optimizer(source, "Adam")} if present == "optimizer" else {})
    path = tmp_path / "absent.ckpt"
    ortho.save_checkpoint(path, **options)
    with pytest.raises(RuntimeError):
        ortho.load_checkpoint(path, model=model, optimizer=optimizer)
    unchanged()


@pytest.mark.parametrize("damage", ["empty", "plain-data", "truncated", "missing-file"])
def test_invalid_file_does_not_change_any_destination_state(
    saved_training_checkpoint, rejection_target, assert_destination_unchanged, tmp_path, damage,
):
    model, optimizer = rejection_target
    unchanged = assert_destination_unchanged(model, optimizer)
    path = tmp_path / "invalid.ckpt"
    if damage == "empty":
        path.write_bytes(b"")
    elif damage == "plain-data":
        path.write_bytes(b"plain text is not a checkpoint")
    elif damage == "truncated":
        raw = saved_training_checkpoint.read_bytes()
        path.write_bytes(raw[:len(raw) // 2])
    with pytest.raises(RuntimeError):
        ortho.load_checkpoint(path, model=model, optimizer=optimizer)
    unchanged()


def test_failed_decode_does_not_invalidate_a_preexisting_saved_value_graph(
    saved_training_checkpoint, checkpoint_archive, tmp_path, assert_tensor,
):
    model = RegressionModel()
    pending = (model.weight * model.weight).sum()
    archive = checkpoint_archive(saved_training_checkpoint)
    archive.manifest["sections"]["metadata"] = {"k": "unsupported"}
    with pytest.raises(RuntimeError):
        ortho.load_checkpoint(archive.write(tmp_path / "graph.ckpt"), model=model)
    pending.backward()
    assert_tensor(model.weight.grad, [[1.], [-0.5]])


def test_failed_encoding_preserves_old_checkpoint_and_source_state(
    training_pair, tmp_path,
):
    from .workload import training_state

    model, optimizer = training_pair()
    advance(model, optimizer, 1)
    path = tmp_path / "replace.ckpt"
    ortho.save_checkpoint(path, model=model, optimizer=optimizer, metadata={"step": 1})
    old_file = path.read_bytes()
    expected_files = set(tmp_path.iterdir())
    advance(model, optimizer, 1)
    before = training_state(model, optimizer)
    with pytest.raises(TypeError):
        ortho.save_checkpoint(path, model=model, optimizer=optimizer,
                              metadata={"step": 2, "late": [0, object()]})
    assert path.read_bytes() == old_file
    assert set(tmp_path.iterdir()) == expected_files
    assert training_state(model, optimizer) == before
    assert ortho.load_checkpoint(path)["step"] == 1


def test_failed_save_at_fresh_path_leaves_no_file_or_temporary(tmp_path):
    path = tmp_path / "not-created.ckpt"
    with pytest.raises(TypeError):
        ortho.save_checkpoint(path, metadata={"unsupported": object()})
    assert list(tmp_path.iterdir()) == []


def test_failed_filesystem_replacement_preserves_directory_and_cleans_temporary(tmp_path):
    target = tmp_path / "existing-directory"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_bytes(b"original contents")
    with pytest.raises(OSError):
        ortho.save_checkpoint(target, metadata={"step": 1})
    assert marker.read_bytes() == b"original contents"
    assert set(tmp_path.iterdir()) == {target}


def test_missing_parent_directory_is_not_created_by_save(tmp_path):
    parent = tmp_path / "absent"
    with pytest.raises(RuntimeError):
        ortho.save_checkpoint(parent / "checkpoint.ckpt", metadata={"step": 1})
    assert not parent.exists()


def test_successful_replacement_loads_new_complete_training_state(training_pair, tmp_path):
    from .workload import training_state

    model, optimizer = training_pair()
    path = tmp_path / "latest.ckpt"
    advance(model, optimizer, 1)
    ortho.save_checkpoint(path, model=model, optimizer=optimizer, metadata={"step": 1})
    old = path.read_bytes()
    advance(model, optimizer, 1)
    ortho.save_checkpoint(path, model=model, optimizer=optimizer, metadata={"step": 2})
    assert path.read_bytes() != old
    assert set(tmp_path.iterdir()) == {path}
    restored, resumed = training_pair()
    assert ortho.load_checkpoint(path, model=restored, optimizer=resumed) == {"step": 2}
    assert training_state(restored, resumed) == training_state(model, optimizer)
