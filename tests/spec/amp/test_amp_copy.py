"""Internal copy semantic boundary: values, identity, validation and versions."""
import pytest
import orthon as ortho


@pytest.mark.parametrize("dtype,values,expected", [
    ("float16", [1.00048828125, 1.00146484375, -2.25], [1., 1.001953125, -2.25]),
    ("bfloat16", [1.00390625, 1.01171875, -2.25], [1., 1.015625, -2.25]),
    ("float32", [1.00048828125, 1.00390625, -2.25], [1.00048828125, 1.00390625, -2.25]),
])
def test_amp_copy_rne_retains_destination_owner_alias_and_metadata(operator_device, typed_tensor, dtype, values, expected):
    source = typed_tensor(values, device=operator_device)
    target = typed_tensor([0.] * 3, dtype=dtype, device=operator_device).requires_grad_(True)
    alias = target.reshape([1, 3])
    impl, version = target._impl_id, target._storage_version
    with ortho.no_grad():
        assert ortho._amp_copy_master_to_model_(target, source) is None
    assert target._impl_id == impl
    assert target._storage_version == alias._storage_version == version + 1
    assert target.tolist() == alias.tolist() == expected
    assert target.dtype == dtype and target.is_leaf and target.requires_grad
    assert target.grad is None


@pytest.mark.parametrize("fault", ["shape", "source-dtype", "destination-dtype", "source-stride", "destination-stride"])
def test_amp_copy_validation_preserves_destination_and_version(typed_tensor, fault):
    source = typed_tensor([[2., 3.], [4., 5.]])
    target = typed_tensor([[1., 1.], [1., 1.]], dtype="float16")
    if fault == "shape":
        source = source.reshape([4])
    elif fault == "source-dtype":
        source = source.cast("float64")
    elif fault == "destination-dtype":
        target = target.cast("int32")
    elif fault == "source-stride":
        source = source.transpose(0, 1)
    else:
        target = target.transpose(0, 1)
    before, version = target.tolist(), target._storage_version
    with pytest.raises(RuntimeError):
        ortho._amp_copy_master_to_model_(target, source)
    assert target.tolist() == before and target._storage_version == version


def test_amp_copy_exact_device_preflight_precedes_any_gpu_launch(cuda_device, typed_tensor):
    source = typed_tensor([1., 2.])
    target = typed_tensor([3., 4.], dtype="float16", device=cuda_device)
    before, version = target.tolist(), target._storage_version
    with pytest.raises(RuntimeError, match="device mismatch"):
        ortho._amp_copy_master_to_model_(target, source)
    assert target.tolist() == before and target._storage_version == version
