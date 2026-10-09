import numpy as np
import pytest
import orthon as ortho


@pytest.mark.parametrize("nan", [False, True], ids=["ties", "nans"])
@pytest.mark.parametrize("dim", [None, 0, 1, -1])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("transpose", [False, True])
def test_max_backward_scatters_to_saved_logical_winner(
    operator_profile, typed_tensor, nan, dim, keepdim, transpose,
):
    device, dtype = operator_profile
    raw = np.array([[1., 7., 7.], [7., 2., 7.]])
    if nan:
        raw[raw == 7.] = np.nan
    leaf = typed_tensor(raw, dtype=dtype, device=device).requires_grad_(True)
    source = leaf.transpose(0, 1) if transpose else leaf
    logical = raw.T if transpose else raw
    source.max(dim=dim, keepdim=keepdim).backward()
    expected = np.zeros(logical.shape)
    if dim is None:
        expected.reshape(-1)[np.argmax(logical)] = 1
    else:
        np.put_along_axis(expected, np.expand_dims(np.argmax(logical, axis=dim), dim), 1., axis=dim)
    if transpose:
        expected = expected.T
    assert leaf.grad.device == leaf.device and leaf.grad.dtype == dtype
    np.testing.assert_array_equal(leaf.grad.tolist(), expected.reshape(-1))


@pytest.mark.parametrize("dim", [None, 0, 1])
def test_max_saved_winner_survives_input_value_mutation(tensor, assert_grad, dim):
    leaf = tensor([[1., 7., 7.], [7., 2., 7.]], requires_grad=True)
    maximum = leaf.max(dim=dim)
    with ortho.no_grad():
        leaf.mul_(-1)
    maximum.backward()
    expected = np.zeros((2, 3))
    if dim is None:
        expected[0, 1] = 1
    elif dim == 0:
        expected[1, 0] = expected[0, 1] = expected[0, 2] = 1
    else:
        expected[0, 1] = expected[1, 0] = 1
    assert_grad(leaf, expected)


def test_max_backward_accepts_strided_upstream(operator_profile, typed_tensor):
    device, dtype = operator_profile
    raw = np.arange(24.).reshape(2, 3, 4)
    leaf = typed_tensor(raw, dtype=dtype, device=device).requires_grad_(True)
    reduced = leaf.max(dim=1)
    weights = typed_tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]], dtype=dtype, device=device)
    (reduced.transpose(0, 1) * weights).sum().backward()
    expected = np.zeros_like(raw)
    expected[:, 2, :] = np.array(weights.tolist()).reshape(4, 2).T
    assert leaf.grad.device == leaf.device
    np.testing.assert_array_equal(leaf.grad.tolist(), expected.reshape(-1))
