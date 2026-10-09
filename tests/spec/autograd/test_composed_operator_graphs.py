import numpy as np

import orthon as ortho


def test_embedding_duplicate_rows_and_matrix_projection_accumulate_to_the_right_owners(
    operator_device, typed_tensor, assert_output,
):
    raw = np.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.]]) / 4
    projection = np.array([[1.], [2.]])
    weight = typed_tensor(raw, device=operator_device).requires_grad_(True)
    matrix = typed_tensor(projection, device=operator_device).requires_grad_(True)
    ids = np.array([[1, 1], [2, 3]])
    indices = typed_tensor(ids, dtype="int64", device=operator_device)
    lookup = ortho.embedding(indices, weight)
    selected = lookup.reshape([4, 2])
    output = selected @ matrix
    (output * output).mean().backward()
    rows = raw[ids.reshape(-1)]
    upstream = 2 * (rows @ projection) / 4
    expected = np.zeros_like(raw)
    np.add.at(expected, ids.reshape(-1), upstream @ projection.T)
    assert_output(weight.grad, expected, dtype="float32", device=operator_device)
    assert_output(matrix.grad, rows.T @ upstream, dtype="float32", device=operator_device)
    assert indices.grad is None and lookup.grad is None and output.grad is None


def test_convolution_relu_pool_and_a_shared_mean_branch_combine_chain_rules(
    operator_device, typed_tensor, assert_output,
):
    raw = np.arange(1, 10).reshape(1, 1, 3, 3)
    image = typed_tensor(raw, device=operator_device).requires_grad_(True)
    kernel = typed_tensor(np.ones((1, 1, 2, 2)), device=operator_device).requires_grad_(True)
    bias = typed_tensor([.5], device=operator_device).requires_grad_(True)
    feature = ortho.relu(ortho.conv2d(image, kernel, bias))
    maximum = ortho.maxpool2d(feature, 2, 2, 2, 2)
    ((maximum * maximum).sum() + feature.mean()).backward()
    upstream = np.full((2, 2), .25)
    upstream[1, 1] += 57
    image_gradient, kernel_gradient = np.zeros((3, 3)), np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            image_gradient[i:i + 2, j:j + 2] += upstream[i, j]
            kernel_gradient += upstream[i, j] * raw[0, 0, i:i + 2, j:j + 2]
    assert_output(image.grad, image_gradient[None, None], dtype="float32", device=operator_device)
    assert_output(kernel.grad, kernel_gradient[None, None], dtype="float32", device=operator_device)
    assert_output(bias.grad, [58.], dtype="float32", device=operator_device)
    assert feature.grad is None and maximum.grad is None


def test_fused_linear_relu_can_feed_a_weighted_nonlinear_loss(
    operator_device, typed_tensor, assert_output,
):
    a, w, b = np.array([[1., -2.], [3., 1.]]), np.array([[2., -1.], [1., 3.]]), np.array([.5, -.5])
    lhs, weight, bias = [typed_tensor(v, device=operator_device).requires_grad_(True) for v in (a, w, b)]
    output = ortho.fused_linear_relu(lhs, weight, bias)
    ((output * output) + output * 3).mean().backward()
    preactivation = a @ w + b
    upstream = np.where(preactivation > 0, 2 * preactivation + 3, 0.) / 4
    assert_output(lhs.grad, upstream @ w.T, dtype="float32", device=operator_device)
    assert_output(weight.grad, a.T @ upstream, dtype="float32", device=operator_device)
    assert_output(bias.grad, upstream.sum(axis=0), dtype="float32", device=operator_device)


def test_shared_softmax_probabilities_combine_two_nonuniform_consumers(
    operator_device, typed_tensor, assert_output,
):
    raw = np.array([[.25, -.5, 1.], [1., .5, -.25]])
    coefficients = np.array([[1., 2., 4.], [3., 5., 7.]])
    leaf = typed_tensor(raw, device=operator_device).requires_grad_(True)
    probabilities = ortho.softmax(leaf * 2 + .5)
    ((probabilities * probabilities).mean() + (probabilities * typed_tensor(coefficients, device=operator_device)).sum()).backward()
    exp = np.exp(raw * 2 - np.max(raw * 2, axis=1, keepdims=True))
    p = exp / exp.sum(axis=1, keepdims=True)
    upstream = 2 * p / 6 + coefficients
    expected = 2 * p * (upstream - (upstream * p).sum(axis=1, keepdims=True))
    assert_output(leaf.grad, expected, dtype="float32", device=operator_device)
    assert probabilities.grad is None
