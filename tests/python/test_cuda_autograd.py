"""
CUDA autograd tests: ReLU backward and reshape backward.
All CUDA tests are skipped automatically when CUDA is not available.
"""
import pytest
import orthon as ortho

torch = pytest.importorskip("torch")


def _cuda_available():
    try:
        ortho.zeros([1]).to("cuda")
        return True
    except RuntimeError:
        return False


cuda_available = _cuda_available()
requires_cuda = pytest.mark.skipif(not cuda_available, reason="CUDA not available")


def assert_close(actual, expected, rel=1e-5, abs_tol=1e-5):
    assert actual == pytest.approx(expected, rel=rel, abs=abs_tol)


# CUDA ReLU backward.

@requires_cuda
def test_cuda_relu_backward_all_positive():
    x = ortho.full([6], 2.0).to("cuda")
    x.requires_grad_(True)
    ortho.relu(x).backward()
    assert x.grad.to("cpu").tolist() == pytest.approx([1.0] * 6)


@requires_cuda
def test_cuda_relu_backward_all_negative():
    x = ortho.full([6], -2.0).to("cuda")
    x.requires_grad_(True)
    ortho.relu(x).backward()
    assert x.grad.to("cpu").tolist() == pytest.approx([0.0] * 6)


@requires_cuda
def test_cuda_relu_backward_mixed():
    vals = [-3.0, -1.0, 0.0, 1.0, 2.0, 5.0]
    x_cpu = ortho.zeros([6])
    for i, v in enumerate(vals):
        x_cpu[i] = v
    x_cuda = x_cpu.to("cuda")
    x_cuda.requires_grad_(True)
    ortho.relu(x_cuda).backward()
    expected = [1.0 if v > 0 else 0.0 for v in vals]
    assert x_cuda.grad.to("cpu").tolist() == pytest.approx(expected)


@requires_cuda
def test_cuda_relu_grad_lives_on_cuda():
    x = ortho.full([8], 1.0).to("cuda")
    x.requires_grad_(True)
    ortho.relu(x).backward()
    assert x.grad.device.is_cuda()


@requires_cuda
def test_cuda_relu_backward_matches_cpu():
    """CUDA relu backward matches CPU relu backward on the same values."""
    x_cpu = ortho.zeros([8])
    for i in range(8):
        x_cpu[i] = float(i - 4)   # [-4, -3, -2, -1, 0, 1, 2, 3]
    x_cuda = x_cpu.to("cuda")

    x_cpu.requires_grad_(True)
    ortho.relu(x_cpu).backward()

    x_cuda.requires_grad_(True)
    ortho.relu(x_cuda).backward()

    assert_close(x_cpu.grad.tolist(), x_cuda.grad.to("cpu").tolist())


@requires_cuda
def test_cuda_relu_backward_matches_torch():
    vals = [-3.0, -1.0, 0.0, 1.0, 2.0, 5.0]
    x_cpu = ortho.zeros([6])
    for i, v in enumerate(vals):
        x_cpu[i] = v

    x_cuda = x_cpu.to("cuda")
    x_cuda.requires_grad_(True)
    ortho.relu(x_cuda).backward()
    ortho_grad = x_cuda.grad.to("cpu").tolist()

    tx = torch.tensor(vals, requires_grad=True)
    torch.relu(tx).sum().backward()
    torch_grad = tx.grad.tolist()

    assert_close(ortho_grad, torch_grad)


@requires_cuda
def test_cuda_relu_accumulates():
    x = ortho.full([4], 2.0).to("cuda")
    x.requires_grad_(True)
    y = ortho.relu(x)
    y.backward(retain_graph=True)
    y.backward()
    assert x.grad.to("cpu").tolist() == pytest.approx([2.0] * 4)


@requires_cuda
def test_cuda_relu_2d_backward():
    """2-D CUDA tensor: backward populates the full gradient tensor."""
    x = ortho.full([4, 8], 1.0).to("cuda")
    x.requires_grad_(True)
    ortho.relu(x).backward()
    assert x.grad.to("cpu").tolist() == pytest.approx([1.0] * 32)


# CUDA reshape backward.

@requires_cuda
def test_cuda_reshape_backward_basic():
    x = ortho.full([6], 1.0).to("cuda")
    x.requires_grad_(True)
    ortho.reshape(x, [2, 3]).backward()
    assert x.grad.to("cpu").tolist() == pytest.approx([1.0] * 6)


@requires_cuda
def test_cuda_reshape_backward_matches_torch():
    vals = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    x_cpu = ortho.zeros([2, 3])
    for i in range(2):
        for j in range(3):
            x_cpu[[i, j]] = vals[i][j]
    x_cuda = x_cpu.to("cuda")
    x_cuda.requires_grad_(True)
    ortho.reshape(x_cuda, [6]).backward()
    ortho_grad = x_cuda.grad.to("cpu").tolist()

    tx = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    tx.reshape(6).sum().backward()
    torch_grad = tx.grad.flatten().tolist()

    assert_close(ortho_grad, torch_grad)


@requires_cuda
def test_cuda_reshape_grad_lives_on_cuda():
    x = ortho.full([6], 1.0).to("cuda")
    x.requires_grad_(True)
    ortho.reshape(x, [2, 3]).backward()
    assert x.grad.device.is_cuda()


# Conv -> ReLU -> MaxPool -> reshape chain.

@requires_cuda
def test_cuda_conv_relu_pool_reshape_backward():
    """
    Backward passes through Conv2d -> ReLU -> MaxPool2d -> reshape without error.
    Gradient existence and device are checked; exact values are not (those are
    covered by individual op tests).
    """
    from orthon.nn import Conv2d, ReLU, MaxPool2d

    N, C_in, H, W = 1, 1, 8, 8
    C_out, K = 2, 3

    conv = Conv2d(C_in, C_out, K, padding=1)
    relu_layer = ReLU()
    pool = MaxPool2d(2, stride=2)

    conv.to("cuda")

    x = ortho.full([N, C_in, H, W], 1.0).to("cuda")
    x.requires_grad_(True)

    out = conv(x)                                          # [N, C_out, H, W]
    out = relu_layer(out)                                  # [N, C_out, H, W]
    out = pool(out)                                        # [N, C_out, H//2, W//2]
    H2, W2 = H // 2, W // 2
    flat = ortho.reshape(out, [N, C_out * H2 * W2])       # flatten
    flat.backward()                                        # seed = ones(flat.shape)

    assert x.grad is not None
    assert x.grad.device.is_cuda()

    weight_grad = conv._parameters['weight'].grad
    bias_grad   = conv._parameters['bias'].grad
    assert weight_grad is not None
    assert bias_grad   is not None
    assert weight_grad.device.is_cuda()
    assert bias_grad.device.is_cuda()


# CUDA add backward.

@requires_cuda
def test_cuda_add_backward_same_shape():
    """add(a, b) backward: same shape, both accumulate grad_output."""
    a = ortho.full([4], 2.0).to("cuda")
    b = ortho.full([4], 3.0).to("cuda")
    a.requires_grad_(True)
    b.requires_grad_(True)
    (a + b).backward()
    assert a.grad.to("cpu").tolist() == pytest.approx([1.0] * 4)
    assert b.grad.to("cpu").tolist() == pytest.approx([1.0] * 4)


@requires_cuda
def test_cuda_add_backward_broadcast_bias():
    """add([N, out], [out]) backward: grad_bias = sum over batch dim."""
    N, out = 4, 3
    a = ortho.full([N, out], 1.0).to("cuda")
    b = ortho.full([out], 1.0).to("cuda")
    a.requires_grad_(True)
    b.requires_grad_(True)
    (a + b).backward()
    # grad_a has shape [N, out], all ones
    assert list(a.grad.shape) == [N, out]
    assert a.grad.to("cpu").tolist() == pytest.approx([1.0] * (N * out))
    # grad_b is unbroadcast of ones([N, out]) → sum over N → [out] each = N
    assert list(b.grad.shape) == [out]
    assert b.grad.to("cpu").tolist() == pytest.approx([float(N)] * out)


@requires_cuda
def test_cuda_add_backward_matches_torch():
    a_cpu = ortho.full([3, 4], 2.0)
    b_cpu = ortho.full([4], 1.0)
    a_cuda = a_cpu.to("cuda")
    b_cuda = b_cpu.to("cuda")
    a_cuda.requires_grad_(True)
    b_cuda.requires_grad_(True)
    (a_cuda + b_cuda).backward()

    ta = torch.full((3, 4), 2.0, requires_grad=True)
    tb = torch.full((4,), 1.0, requires_grad=True)
    (ta + tb).sum().backward()

    assert_close(a_cuda.grad.to("cpu").tolist(), ta.grad.flatten().tolist())
    assert_close(b_cuda.grad.to("cpu").tolist(), tb.grad.tolist())


@requires_cuda
def test_cuda_add_grad_lives_on_cuda():
    a = ortho.full([8], 1.0).to("cuda")
    b = ortho.full([8], 2.0).to("cuda")
    a.requires_grad_(True)
    b.requires_grad_(True)
    (a + b).backward()
    assert a.grad.device.is_cuda()
    assert b.grad.device.is_cuda()


# CUDA mul backward.

@requires_cuda
def test_cuda_mul_backward_basic():
    a = ortho.full([4], 2.0).to("cuda")
    b = ortho.full([4], 3.0).to("cuda")
    a.requires_grad_(True)
    b.requires_grad_(True)
    (a * b).backward()
    # grad_a = grad_output * b = 1 * 3 = 3; grad_b = 1 * 2 = 2
    assert a.grad.to("cpu").tolist() == pytest.approx([3.0] * 4)
    assert b.grad.to("cpu").tolist() == pytest.approx([2.0] * 4)


@requires_cuda
def test_cuda_mul_backward_matches_torch():
    vals_a = [1.0, -2.0, 3.0, 0.5]
    vals_b = [4.0,  1.0, -1.0, 2.0]

    a_cpu = ortho.zeros([4])
    b_cpu = ortho.zeros([4])
    for i, (va, vb) in enumerate(zip(vals_a, vals_b)):
        a_cpu[i] = va
        b_cpu[i] = vb

    a_cuda = a_cpu.to("cuda")
    b_cuda = b_cpu.to("cuda")
    a_cuda.requires_grad_(True)
    b_cuda.requires_grad_(True)
    (a_cuda * b_cuda).backward()

    ta = torch.tensor(vals_a, requires_grad=True)
    tb = torch.tensor(vals_b, requires_grad=True)
    (ta * tb).sum().backward()

    assert_close(a_cuda.grad.to("cpu").tolist(), ta.grad.tolist())
    assert_close(b_cuda.grad.to("cpu").tolist(), tb.grad.tolist())


@requires_cuda
def test_cuda_mul_grad_lives_on_cuda():
    a = ortho.full([6], 2.0).to("cuda")
    b = ortho.full([6], 3.0).to("cuda")
    a.requires_grad_(True)
    b.requires_grad_(True)
    (a * b).backward()
    assert a.grad.device.is_cuda()
    assert b.grad.device.is_cuda()


# CUDA matmul backward.

@requires_cuda
def test_cuda_matmul_backward_square():
    """(N,K) @ (K,N): square, both inputs need grad."""
    A = ortho.full([4, 4], 1.0).to("cuda")
    B = ortho.full([4, 4], 2.0).to("cuda")
    A.requires_grad_(True)
    B.requires_grad_(True)
    (A @ B).backward()
    # grad_A = ones([4,4]) @ B.T = ones([4,4]) @ full([4,4],2) = full([4,4], 8)
    assert A.grad.to("cpu").tolist() == pytest.approx([8.0] * 16)
    # grad_B = A.T @ ones([4,4]) = full([4,4],1) @ ones([4,4]) = full([4,4], 4)
    assert B.grad.to("cpu").tolist() == pytest.approx([4.0] * 16)


@requires_cuda
def test_cuda_matmul_backward_rectangular():
    """(2,3) @ (3,4): rectangular shapes."""
    A = ortho.full([2, 3], 1.0).to("cuda")
    B = ortho.full([3, 4], 2.0).to("cuda")
    A.requires_grad_(True)
    B.requires_grad_(True)
    (A @ B).backward()
    # grad_A = ones([2,4]) @ B.T = ones([2,4]) @ full([4,3],2) = full([2,3], 8)
    assert A.grad.to("cpu").tolist() == pytest.approx([8.0] * 6)
    # grad_B = A.T @ ones([2,4]) = full([3,2],1) @ ones([2,4]) = full([3,4], 2)
    assert B.grad.to("cpu").tolist() == pytest.approx([2.0] * 12)


@requires_cuda
def test_cuda_matmul_backward_matches_torch():
    A_vals = [float(i + 1) for i in range(6)]   # 1..6 in [2,3]
    B_vals = [float(i + 1) for i in range(12)]  # 1..12 in [3,4]

    A_cpu = ortho.zeros([2, 3])
    B_cpu = ortho.zeros([3, 4])
    for i in range(6):
        A_cpu[[i // 3, i % 3]] = A_vals[i]
    for i in range(12):
        B_cpu[[i // 4, i % 4]] = B_vals[i]

    A_cuda = A_cpu.to("cuda")
    B_cuda = B_cpu.to("cuda")
    A_cuda.requires_grad_(True)
    B_cuda.requires_grad_(True)
    (A_cuda @ B_cuda).backward()

    tA = torch.tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
    tB = torch.tensor([[float(i+1) for i in range(4)],
                        [float(i+5) for i in range(4)],
                        [float(i+9) for i in range(4)]], requires_grad=True)
    (tA @ tB).sum().backward()

    assert_close(A_cuda.grad.to("cpu").tolist(), tA.grad.flatten().tolist(),
                 rel=1e-4, abs_tol=1e-4)
    assert_close(B_cuda.grad.to("cpu").tolist(), tB.grad.flatten().tolist(),
                 rel=1e-4, abs_tol=1e-4)


@requires_cuda
def test_cuda_matmul_grad_lives_on_cuda():
    A = ortho.full([3, 4], 1.0).to("cuda")
    B = ortho.full([4, 5], 1.0).to("cuda")
    A.requires_grad_(True)
    B.requires_grad_(True)
    (A @ B).backward()
    assert A.grad.device.is_cuda()
    assert B.grad.device.is_cuda()


# CUDA Linear backward.

@requires_cuda
def test_cuda_linear_backward_exists():
    """Linear(x @ weight + bias): all three gradients exist and live on CUDA."""
    from orthon.nn import Linear
    N, in_f, out_f = 4, 8, 3
    layer = Linear(in_f, out_f)
    layer.to("cuda")

    x = ortho.full([N, in_f], 1.0).to("cuda")
    x.requires_grad_(True)
    layer(x).backward()

    assert x.grad is not None and x.grad.device.is_cuda()
    assert layer.weight.grad is not None and layer.weight.grad.device.is_cuda()
    assert layer.bias.grad   is not None and layer.bias.grad.device.is_cuda()


@requires_cuda
def test_cuda_linear_bias_grad_shape():
    """bias grad has shape [out_features], not [N, out_features]."""
    from orthon.nn import Linear
    N, in_f, out_f = 5, 4, 6
    layer = Linear(in_f, out_f)
    layer.to("cuda")

    x = ortho.full([N, in_f], 1.0).to("cuda")
    x.requires_grad_(True)
    layer(x).backward()

    assert list(layer.bias.grad.shape) == [out_f]


@requires_cuda
def test_cuda_linear_backward_matches_torch():
    """Linear backward: Orthon CUDA matches PyTorch CPU on same values."""
    ortho.manual_seed(42)          # framework RNG, not Python's
    N, in_f, out_f = 3, 4, 5

    from orthon.nn import Linear
    layer = Linear(in_f, out_f)
    # Extract weight/bias values to build identical PyTorch layer
    w_vals = layer.weight.tolist()
    b_vals = layer.bias.tolist()

    layer.to("cuda")
    x_vals = [float(i + 1) for i in range(N * in_f)]

    x_cpu = ortho.zeros([N, in_f])
    for i in range(N * in_f):
        x_cpu[[i // in_f, i % in_f]] = x_vals[i]
    x_cuda = x_cpu.to("cuda")
    x_cuda.requires_grad_(True)
    layer(x_cuda).backward()

    # PyTorch reference: tolist() is flat, so build 2D leaf tensors directly.
    w_2d = [w_vals[i * out_f:(i + 1) * out_f] for i in range(in_f)]
    tw = torch.tensor(w_2d, dtype=torch.float32, requires_grad=True)  # (in_f, out_f)
    tb = torch.tensor(b_vals, dtype=torch.float32, requires_grad=True)
    x_2d = [x_vals[i * in_f:(i + 1) * in_f] for i in range(N)]
    tx = torch.tensor(x_2d, dtype=torch.float32, requires_grad=True)  # (N, in_f) leaf
    (tx @ tw + tb).sum().backward()

    assert_close(x_cuda.grad.to("cpu").tolist(), tx.grad.flatten().tolist(),
                 rel=1e-4, abs_tol=1e-4)
    assert_close(layer.weight.grad.to("cpu").tolist(), tw.grad.flatten().tolist(),
                 rel=1e-4, abs_tol=1e-4)
    assert_close(layer.bias.grad.to("cpu").tolist(), tb.grad.tolist(),
                 rel=1e-4, abs_tol=1e-4)


# ---- Full CIFAR-like backward: conv -> relu -> pool -> reshape -> Linear -

@requires_cuda
def test_cuda_full_graph_conv_to_linear():
    """
    Backward through Conv2d -> ReLU -> MaxPool2d -> reshape -> Linear.
    Checks all gradients exist and live on CUDA; does not check exact values.
    """
    from orthon.nn import Conv2d, ReLU, MaxPool2d, Linear

    N, C_in, H, W = 2, 1, 8, 8
    C_out, K, pool_k = 4, 3, 2
    H2 = H // pool_k
    W2 = W // pool_k
    flat_features = C_out * H2 * W2
    num_classes = 5

    conv  = Conv2d(C_in, C_out, K, padding=1).to("cuda")
    relu  = ReLU()
    pool  = MaxPool2d(pool_k, stride=pool_k)
    fc    = Linear(flat_features, num_classes).to("cuda")

    x = ortho.full([N, C_in, H, W], 0.5).to("cuda")
    x.requires_grad_(True)

    out = conv(x)
    out = relu(out)
    out = pool(out)
    out = ortho.reshape(out, [N, flat_features])
    out = fc(out)
    out.backward()     # seed = ones([N, num_classes])

    # Input gradient
    assert x.grad is not None
    assert x.grad.device.is_cuda()
    assert list(x.grad.shape) == [N, C_in, H, W]

    # Conv weights
    assert conv._parameters['weight'].grad is not None
    assert conv._parameters['bias'].grad   is not None

    # FC weights
    assert fc.weight.grad is not None and list(fc.weight.grad.shape) == [flat_features, num_classes]
    assert fc.bias.grad   is not None and list(fc.bias.grad.shape)   == [num_classes]


# CUDA cross-entropy backward.

@requires_cuda
def test_cuda_cross_entropy_forward_matches_torch():
    """CUDA cross-entropy forward matches PyTorch F.cross_entropy."""
    import torch.nn.functional as F

    logits_vals = [[2.0, 1.0, 0.5], [0.5, 1.5, 2.0], [1.0, 1.0, 1.0]]
    targets_vals = [0, 2, 1]

    # Orthon
    x_cpu = ortho.zeros([3, 3])
    for i in range(3):
        for j in range(3):
            x_cpu[[i, j]] = logits_vals[i][j]
    x_cuda = x_cpu.to("cuda")
    loss_d = ortho.cross_entropy(x_cuda, targets_vals)

    # PyTorch
    logits_t = torch.tensor(logits_vals, dtype=torch.float32)
    targets_t = torch.tensor(targets_vals, dtype=torch.long)
    loss_t = F.cross_entropy(logits_t, targets_t)

    assert_close([loss_d.tolist()[0]], [loss_t.item()], rel=1e-4, abs_tol=1e-4)


@requires_cuda
def test_cuda_cross_entropy_backward_basic():
    """CUDA cross-entropy backward produces a gradient of the right shape on CUDA."""
    logits_vals = [[2.0, 1.0, 0.5], [0.5, 1.5, 2.0]]
    targets_vals = [0, 2]

    x_cpu = ortho.zeros([2, 3])
    for i in range(2):
        for j in range(3):
            x_cpu[[i, j]] = logits_vals[i][j]
    x_cuda = x_cpu.to("cuda")
    x_cuda.requires_grad_(True)

    loss = ortho.cross_entropy(x_cuda, targets_vals)
    loss.backward()

    assert x_cuda.grad is not None
    assert x_cuda.grad.device.is_cuda()
    assert list(x_cuda.grad.shape) == [2, 3]


@requires_cuda
def test_cuda_cross_entropy_backward_matches_torch():
    """CUDA cross-entropy grad_logits matches PyTorch to 1e-4."""
    import torch.nn.functional as F

    logits_vals = [[2.0, 1.0, 0.5], [0.5, 1.5, 2.0], [1.0, 2.0, 0.0]]
    targets_vals = [0, 2, 1]
    B, C = 3, 3

    # Orthon CUDA
    x_cpu = ortho.zeros([B, C])
    for i in range(B):
        for j in range(C):
            x_cpu[[i, j]] = logits_vals[i][j]
    x_cuda = x_cpu.to("cuda")
    x_cuda.requires_grad_(True)
    ortho.cross_entropy(x_cuda, targets_vals).backward()
    dal_grad = x_cuda.grad.to("cpu").tolist()

    # PyTorch reference
    logits_t = torch.tensor(logits_vals, dtype=torch.float32, requires_grad=True)
    targets_t = torch.tensor(targets_vals, dtype=torch.long)
    F.cross_entropy(logits_t, targets_t).backward()
    ref_grad = logits_t.grad.flatten().tolist()

    assert_close(dal_grad, ref_grad, rel=1e-4, abs_tol=1e-4)


@requires_cuda
def test_cuda_linear_cross_entropy_backward():
    """Linear -> CrossEntropy backward: grad_weight, grad_bias, grad_x all on CUDA."""
    import torch.nn.functional as F
    from orthon.nn import Linear

    N, in_f, out_f = 4, 6, 3
    layer = Linear(in_f, out_f)
    w_vals = layer.weight.tolist()  # flat (in_f * out_f,)
    b_vals = layer.bias.tolist()    # flat (out_f,)
    layer.to("cuda")

    x_vals = [float(i + 1) for i in range(N * in_f)]
    x_cpu = ortho.zeros([N, in_f])
    for i in range(N * in_f):
        x_cpu[[i // in_f, i % in_f]] = x_vals[i]
    x_cuda = x_cpu.to("cuda")
    x_cuda.requires_grad_(True)

    targets = [i % out_f for i in range(N)]
    logits = layer(x_cuda)
    ortho.cross_entropy(logits, targets).backward()

    # Shapes and devices
    assert x_cuda.grad is not None and x_cuda.grad.device.is_cuda()
    assert list(x_cuda.grad.shape) == [N, in_f]
    assert layer.weight.grad is not None and list(layer.weight.grad.shape) == [in_f, out_f]
    assert layer.bias.grad   is not None and list(layer.bias.grad.shape)   == [out_f]

    # Compare to PyTorch
    w_2d = [w_vals[i * out_f:(i + 1) * out_f] for i in range(in_f)]
    x_2d = [x_vals[i * in_f:(i + 1) * in_f] for i in range(N)]
    tw = torch.tensor(w_2d, dtype=torch.float32, requires_grad=True)
    tb = torch.tensor(b_vals, dtype=torch.float32, requires_grad=True)
    tx = torch.tensor(x_2d, dtype=torch.float32, requires_grad=True)
    tt = torch.tensor(targets, dtype=torch.long)
    F.cross_entropy(tx @ tw + tb, tt).backward()

    assert_close(x_cuda.grad.to("cpu").tolist(), tx.grad.flatten().tolist(), rel=1e-3, abs_tol=1e-3)
    assert_close(layer.weight.grad.to("cpu").tolist(), tw.grad.flatten().tolist(), rel=1e-3, abs_tol=1e-3)
    assert_close(layer.bias.grad.to("cpu").tolist(), tb.grad.tolist(), rel=1e-3, abs_tol=1e-3)


@requires_cuda
def test_cuda_full_graph_with_cross_entropy():
    """
    Backward through Conv2d -> ReLU -> MaxPool2d -> reshape -> Linear -> CrossEntropy.
    Checks all gradients exist and live on CUDA.
    """
    from orthon.nn import Conv2d, ReLU, MaxPool2d, Linear

    N, C_in, H, W = 2, 1, 8, 8
    C_out, K, pool_k = 4, 3, 2
    num_classes = 5
    H2 = H // pool_k
    W2 = W // pool_k
    flat_features = C_out * H2 * W2

    conv = Conv2d(C_in, C_out, K, padding=1).to("cuda")
    relu = ReLU()
    pool = MaxPool2d(pool_k, stride=pool_k)
    fc   = Linear(flat_features, num_classes).to("cuda")

    x = ortho.full([N, C_in, H, W], 0.5).to("cuda")
    x.requires_grad_(True)
    targets = [i % num_classes for i in range(N)]

    out = conv(x)
    out = relu(out)
    out = pool(out)
    out = ortho.reshape(out, [N, flat_features])
    out = fc(out)
    loss = ortho.cross_entropy(out, targets)
    assert loss.device.is_cuda()
    loss.backward()

    assert x.grad is not None and x.grad.device.is_cuda()
    assert list(x.grad.shape) == [N, C_in, H, W]
    assert conv._parameters['weight'].grad is not None
    assert fc.weight.grad is not None
    assert fc.bias.grad   is not None
