"""Reductions define first-NaN extrema; finite ties use logical index order."""
import numpy as np
import pytest
import orthon as ortho

PATTERNS = [
    pytest.param([1, 7, 2, 3, 4, 5], id="unique"),
    pytest.param([1, 7, 7, 7, 2, 7], id="ties"),
    pytest.param([-np.inf, 1, np.inf, np.inf, 2, -np.inf], id="positive-infinity"),
    pytest.param([-np.inf] * 6, id="all-negative-infinity"),
    pytest.param([1, np.nan, 7, 4, 2, 6], id="mixed-nan"),
    pytest.param([1, np.nan, np.nan, np.nan, 2, 6], id="multiple-nans"),
    pytest.param([-0., +0., -0., +0., -0., +0.], id="signed-zero"),
]


@pytest.mark.parametrize("values", PATTERNS)
@pytest.mark.parametrize("layout", ["dense", "transpose", "offset"])
@pytest.mark.parametrize("dim", [None, 0, 1, -1])
@pytest.mark.parametrize("keepdim", [False, True])
def test_extrema_select_first_logical_winner(
    operator_profile, typed_tensor, values, layout, dim, keepdim,
):
    device, dtype = operator_profile
    logical = np.array(values).reshape(2, 3)
    if layout == "offset":
        storage = np.full((4, 5), -13.)
        storage[1:3, 1:4] = logical
        source = typed_tensor(storage, dtype=dtype, device=device).slice(0, 1, 3).slice(1, 1, 4)
    else:
        source = typed_tensor(logical, dtype=dtype, device=device)
        if layout == "transpose":
            source = source.transpose(0, 1)
            logical = logical.T
    source.requires_grad_(True)
    indices = np.argmax(logical, axis=dim)
    if dim is None:
        selected = logical.reshape(-1)[indices]
    else:
        selected = np.take_along_axis(logical, np.expand_dims(indices, dim), axis=dim).squeeze(dim)
    if keepdim:
        shape = (1,) * logical.ndim if dim is None else tuple(1 if i == dim % logical.ndim else n for i, n in enumerate(logical.shape))
        indices = np.reshape(indices, shape)
        selected = np.reshape(selected, shape)
    maximum = source.max(dim=dim, keepdim=keepdim)
    argmax = source.argmax(dim=dim, keepdim=keepdim)
    assert maximum.shape == np.shape(selected)
    assert maximum.dtype == dtype and maximum.device == source.device
    assert argmax.shape == np.shape(indices)
    assert argmax.dtype == "int64" and argmax.device == source.device
    assert not argmax.requires_grad and argmax.grad is None
    assert argmax.tolist() == np.asarray(indices).reshape(-1).tolist()
    actual = np.array(maximum.tolist())
    expected = np.asarray(selected).reshape(-1)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(actual[~np.isnan(expected)], expected[~np.isnan(expected)])
    zero = expected == 0
    np.testing.assert_array_equal(np.signbit(actual[zero]), np.signbit(expected[zero]))


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("nan", [False, True], ids=["ties", "nans"])
def test_cuda_global_extrema_merge_partial_blocks_by_logical_index(cuda_device, typed_tensor, dtype, nan):
    # More than 1024 blocks' worth also exercises each thread's grid-stride loop.
    values = np.full(262145, -2., dtype=np.float32)
    positions = [13, 256, 65539, 262144]
    values[positions] = np.nan if nan else 7.
    source = typed_tensor(values, dtype=dtype, device=cuda_device)
    for _ in range(5):
        assert source.argmax().item() == positions[0]
        selected = source.max().item()
        assert np.isnan(selected) if nan else selected == 7.
