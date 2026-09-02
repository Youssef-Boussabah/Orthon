import orthon as ortho


def test_default_requires_grad_is_false():
    t = ortho.ones([2, 3])
    assert t.requires_grad is False


def test_default_is_leaf_is_true():
    t = ortho.ones([2, 3])
    assert t.is_leaf is True


def test_requires_grad_inplace_returns_same_object():
    t = ortho.ones([2, 3])
    ret = t.requires_grad_(True)
    assert ret is t
    assert t.requires_grad is True


def test_requires_grad_inplace_false():
    t = ortho.ones([2, 3])
    t.requires_grad_(True)
    t.requires_grad_(False)
    assert t.requires_grad is False


def test_add_propagates_requires_grad():
    a = ortho.ones([2, 3])
    b = ortho.ones([2, 3])
    a.requires_grad_(True)
    c = a + b
    assert c.requires_grad is True
    assert c.is_leaf is False


def test_add_no_grad_stays_leaf():
    a = ortho.ones([2, 3])
    b = ortho.ones([2, 3])
    c = a + b
    assert c.requires_grad is False
    assert c.is_leaf is True


def test_add_both_inputs_require_grad():
    a = ortho.ones([2, 3])
    b = ortho.ones([2, 3])
    a.requires_grad_(True)
    b.requires_grad_(True)
    c = a + b
    assert c.requires_grad is True
    assert c.is_leaf is False


def test_mul_propagates_requires_grad():
    a = ortho.ones([2, 3])
    b = ortho.ones([2, 3])
    b.requires_grad_(True)
    c = a * b
    assert c.requires_grad is True
    assert c.is_leaf is False


def test_mul_no_grad_stays_leaf():
    a = ortho.ones([2, 3])
    b = ortho.ones([2, 3])
    c = a * b
    assert c.requires_grad is False
    assert c.is_leaf is True


def test_matmul_propagates_requires_grad():
    a = ortho.ones([2, 3])
    b = ortho.ones([3, 4])
    a.requires_grad_(True)
    c = a @ b
    assert c.requires_grad is True
    assert c.is_leaf is False


def test_matmul_no_grad_stays_leaf():
    a = ortho.ones([2, 3])
    b = ortho.ones([3, 4])
    c = a @ b
    assert c.requires_grad is False
    assert c.is_leaf is True


def test_user_set_requires_grad_remains_leaf():
    t = ortho.ones([2, 3])
    t.requires_grad_(True)
    assert t.is_leaf is True


def test_grad_is_none_by_default():
    t = ortho.ones([2, 3])
    assert t.grad is None


def test_zero_grad_clears_grad():
    t = ortho.ones([2, 3])
    t.requires_grad_(True)
    g = ortho.zeros([2, 3])
    t.set_grad(g)
    assert t.grad is not None
    t.zero_grad()
    assert t.grad is None


def test_chained_ops_propagate_requires_grad():
    a = ortho.ones([2, 3])
    b = ortho.ones([2, 3])
    a.requires_grad_(True)
    c = a + b
    d = c * b
    assert d.requires_grad is True
    assert d.is_leaf is False
