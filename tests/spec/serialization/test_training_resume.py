import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

import orthon as ortho

from .workload import RegressionModel, advance, make_optimizer, training_state


def test_workload_first_sgd_step_has_hand_computed_values(assert_tensor):
    model = RegressionModel(dtype="float64")
    optimizer = make_optimizer(model, "SGD")
    assert advance(model, optimizer, 1) == [0.0390625]
    assert_tensor(model.weight, [[0.53125], [-0.24609375]])
    assert_tensor(model.head.bias, [0.140625])
    assert_tensor(model.spare, [5.])


@pytest.mark.parametrize("prefix_steps", [0, 1, 3])
def test_checkpoint_resume_matches_each_remaining_loss_and_final_state(
    training_pair, tmp_path, prefix_steps,
):
    uninterrupted, optimizer = training_pair()
    advance(uninterrupted, optimizer, prefix_steps)
    path = tmp_path / "split.ckpt"
    assert ortho.save_checkpoint(path, model=uninterrupted, optimizer=optimizer,
                                 metadata={"completed_steps": prefix_steps}) is None
    expected_losses = advance(uninterrupted, optimizer, 3)
    expected_state = training_state(uninterrupted, optimizer)
    resumed, restored_optimizer = training_pair(lr=0.5)
    held = list(resumed.parameters())
    metadata = ortho.load_checkpoint(path, model=resumed, optimizer=restored_optimizer)
    assert metadata == {"completed_steps": prefix_steps}
    assert all(current is original for current, original in zip(resumed.parameters(), held))
    assert all(p.grad is None for p in resumed.parameters())
    assert advance(resumed, restored_optimizer, 3) == expected_losses
    assert training_state(resumed, restored_optimizer) == expected_state


@pytest.mark.parametrize("name", ["SGD", "Adam"])
def test_resume_works_after_save_process_exits(name, tmp_path):
    checkpoint = tmp_path / "process.ckpt"
    script = textwrap.dedent("""\
        import json, sys
        from pathlib import Path
        import orthon as ortho
        from workload import RegressionModel, advance, make_optimizer, training_state
        phase, name, checkpoint = sys.argv[1:]
        model = RegressionModel()
        optimizer = make_optimizer(model, name, lr=0.125 if phase == 'save' else 0.5)
        if phase == 'save':
            advance(model, optimizer, 2)
            ortho.save_checkpoint(checkpoint, model=model, optimizer=optimizer,
                                  metadata={'completed_steps': 2})
        else:
            metadata = ortho.load_checkpoint(checkpoint, model=model, optimizer=optimizer)
            losses = advance(model, optimizer, 3)
            result = {'metadata': metadata, 'losses': losses,
                      'state': training_state(model, optimizer)}
            Path(checkpoint + '.json').write_text(json.dumps(result))
        """)
    env = dict(os.environ)
    helper_path = str(Path(__file__).parent)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (helper_path, env.get("PYTHONPATH"))))
    for phase in ("save", "resume"):
        result = subprocess.run([sys.executable, "-c", script, phase, name, str(checkpoint)],
                                cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
    restored = json.loads(Path(str(checkpoint) + ".json").read_text())
    reference = RegressionModel()
    optimizer = make_optimizer(reference, name)
    advance(reference, optimizer, 2)
    assert restored["metadata"] == {"completed_steps": 2}
    assert restored["losses"] == advance(reference, optimizer, 3)
    assert restored["state"] == training_state(reference, optimizer)


def test_saved_file_branches_have_independent_model_and_adam_state(
    operator_device, tmp_path,
):
    source = RegressionModel(device=operator_device)
    optimizer = make_optimizer(source, "Adam")
    advance(source, optimizer, 2)
    path = tmp_path / "branches.ckpt"
    ortho.save_checkpoint(path, model=source, optimizer=optimizer)
    before = training_state(source, optimizer)
    left, right = (RegressionModel(device=operator_device) for _ in range(2))
    a, b = make_optimizer(left, "Adam"), make_optimizer(right, "Adam")
    ortho.load_checkpoint(path, model=left, optimizer=a)
    ortho.load_checkpoint(path, model=right, optimizer=b)
    advance(left, a, 1)
    assert training_state(source, optimizer) == before
    assert training_state(right, b) == before
    assert advance(right, b, 1) == advance(source, optimizer, 1)
    assert training_state(left, a) == training_state(right, b) == training_state(source, optimizer)


def test_model_first_explicit_remap_then_optimizer_restore_can_resume(
    resume_case, training_pair, tmp_path,
):
    name, device, dtype = resume_case
    source, optimizer = training_pair()
    advance(source, optimizer, 2)
    path = tmp_path / "remapped.ckpt"
    ortho.save_checkpoint(path, model=source, optimizer=optimizer)
    expected = advance(source, optimizer, 2)
    destination = RegressionModel(dtype)
    # Bind the optimizer after explicit placement has established the destination tensors.
    ortho.load_checkpoint(path, model=destination, map_location=device)
    restored = make_optimizer(destination, name, lr=0.5)
    ortho.load_checkpoint(path, optimizer=restored)
    assert advance(destination, restored, 2) == expected
    assert training_state(destination, restored) == training_state(source, optimizer)


def test_separate_model_and_optimizer_files_can_resume_one_training_boundary(
    training_pair, tmp_path,
):
    model, optimizer = training_pair()
    advance(model, optimizer, 2)
    model_path, optimizer_path = tmp_path / "model.ckpt", tmp_path / "optimizer.ckpt"
    ortho.save_state_dict(model, model_path)
    ortho.save_checkpoint(optimizer_path, optimizer=optimizer, metadata={"step": 2})
    expected = advance(model, optimizer, 2)
    restored, resumed = training_pair(lr=0.5)
    ortho.load_state_dict(restored, model_path)
    assert ortho.load_checkpoint(optimizer_path, optimizer=resumed) == {"step": 2}
    assert advance(restored, resumed, 2) == expected
    assert training_state(restored, resumed) == training_state(model, optimizer)
