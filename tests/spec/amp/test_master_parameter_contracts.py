import orthon as ortho
from orthon.amp import MixedPrecisionOptimizer
from orthon.nn import Module, Parameter
from orthon.optim import SGD


def test_public_master_state_describes_the_optimization_leaves(amp_model, amp_storage_case, assert_output):
    device, _ = amp_storage_case
    optimizer = MixedPrecisionOptimizer(amp_model, SGD, lr=.25)
    parameters = list(amp_model.parameters())
    state = optimizer.state_dict()
    assert state["version"] == 1
    assert state["orig_dtypes"] == [p.dtype for p in parameters]
    assert len(state["masters"]) == len(parameters)
    for parameter, master in zip(parameters, state["masters"]):
        assert_output(master, parameter.tolist() if len(parameter.shape) == 1 else [[1., 2.], [3., 4.]],
                      dtype="float32", device=device)
        assert master.shape == parameter.shape
        assert master.requires_grad and master.is_leaf and master.grad is None
        assert parameter.requires_grad and parameter.is_leaf
        if parameter.dtype == "float32":
            assert master is parameter
        else:
            assert master is not parameter


def test_model_and_master_mutation_ownership_follows_precision(
    amp_model, amp_storage_case, assert_output,
):
    device, dtype = amp_storage_case
    optimizer = MixedPrecisionOptimizer(amp_model, SGD, lr=.25)
    master = optimizer.state_dict()["masters"][0]
    changed = amp_model.state_dict()
    changed["weight"] = ortho.full([2, 2], 9, dtype=dtype).to(device)
    amp_model.load_state_dict(changed)
    assert_output(master, [[9., 9.], [9., 9.]] if dtype == "float32" else [[1., 2.], [3., 4.]],
                  dtype="float32", device=device)
    master.fill_(.75)
    assert_output(amp_model.weight, [[.75, .75], [.75, .75]] if dtype == "float32" else [[9., 9.], [9., 9.]],
                  dtype=dtype, device=device)
    assert optimizer.sync_to_model() is None
    assert_output(amp_model.weight, [[.75, .75], [.75, .75]], dtype=dtype, device=device)
    assert amp_model.weight.requires_grad and amp_model.weight.is_leaf
    assert amp_model.weight.grad is None


def test_mixed_dtype_model_has_separate_low_precision_and_shared_fp32_owners(
    amp_storage_case, typed_tensor, assert_output,
):
    device, dtype = amp_storage_case
    model = Module()
    model.low = Parameter(typed_tensor([1., 2.], dtype=dtype, device=device))
    model.full = Parameter(typed_tensor([3., 4.], device=device))
    optimizer = MixedPrecisionOptimizer(model, SGD, lr=.25)
    low, full = optimizer.state_dict()["masters"]
    assert full is model.full
    low.fill_(.5)
    full.fill_(1.5)
    assert optimizer.sync_to_model() is None
    assert_output(model.low, [.5, .5], dtype=dtype, device=device)
    assert_output(model.full, [1.5, 1.5], dtype="float32", device=device)
    assert model.full is full


def test_cuda_sync_preserves_registered_model_owners_and_existing_gradients(
    cuda_device, typed_tensor, model_gradient, assert_output,
):
    model = Module()
    model.weight = Parameter(typed_tensor([[1., 2.], [3., 4.]], dtype="float16", device=cuda_device))
    optimizer = MixedPrecisionOptimizer(model, SGD, lr=.25)
    owner = model.weight
    model_gradient(owner, [[2., 4.], [6., 8.]])
    optimizer.state_dict()["masters"][0].fill_(.75)
    for _ in range(2):
        assert optimizer.sync_to_model() is None
        assert model.weight is owner
        assert next(model.parameters()) is owner
        assert_output(owner, [[.75, .75], [.75, .75]], dtype="float16", device=cuda_device)
        assert_output(owner.grad, [[2., 4.], [6., 8.]], dtype="float16", device=cuda_device)


def test_small_updates_accumulate_in_fp32_even_below_model_precision(
    amp_model, amp_storage_case, model_gradient,
):
    import pytest
    _, dtype = amp_storage_case
    optimizer = MixedPrecisionOptimizer(amp_model, SGD, lr=.125)
    for _ in range(3):
        optimizer.zero_grad()
        model_gradient(amp_model.weight, [[2**-10] * 2] * 2)
        optimizer.step()
    master = optimizer.state_dict()["masters"][0]
    assert master.tolist() == pytest.approx([x - 3 * 2**-13 for x in (1., 2., 3., 4.)], abs=1e-7)
    if dtype != "float32":
        assert amp_model.weight.tolist()[0] != master.tolist()[0]
