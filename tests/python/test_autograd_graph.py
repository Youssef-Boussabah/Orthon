import orthon as ortho


def test_add_creates_grad_fn():
    a = ortho.ones([2, 3])
    b = ortho.ones([2, 3])

    a.requires_grad_(True)
    c = a + b

    assert c.requires_grad is True
    assert c.is_leaf is False
    assert c._has_grad_fn is True


def test_no_grad_no_graph():
    a = ortho.ones([2, 3])
    b = ortho.ones([2, 3])

    c = a + b

    assert c.requires_grad is False
    assert c._has_grad_fn is False


def test_both_inputs_tracked():
    a = ortho.ones([2, 3])
    b = ortho.ones([2, 3])
    a.requires_grad_(True)
    b.requires_grad_(True)
    c = a + b

    assert c._has_grad_fn is True


def test_mul_creates_grad_fn():
    a = ortho.ones([2, 3])
    b = ortho.ones([2, 3])
    a.requires_grad_(True)
    c = a * b

    assert c.requires_grad is True
    assert c.is_leaf is False
    assert c._has_grad_fn is True


def test_leaf_tensors_have_no_grad_fn():
    t = ortho.ones([2, 3])
    t.requires_grad_(True)

    assert t.is_leaf is True
    assert t._has_grad_fn is False
