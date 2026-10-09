"""Shared storage versions and selective saved-value mutation checks.

Value-sensitive Nodes reject stale inputs before gradient publication;
shape-only derivatives permit mutations. Aliases share exactly-once version bumps."""
from tests.support.capabilities import orthon_cuda_available
import numpy as np
import pytest

import orthon as ortho

CUDA = orthon_cuda_available()
DEVICES = ["cpu"] + (["cuda"] if CUDA else [])
requires_cuda = pytest.mark.skipif(not CUDA, reason="CUDA not available")

# Every stale-storage rejection funnels through Node::validate_saved_versions,
# so the message shape is uniform and worth pinning: op, role, both versions.
STALE = r"modified in-place"


def _t(values, device="cpu", dtype=np.float32, grad=False):
    t = ortho.from_numpy(np.asarray(values, dtype=dtype))
    if device != "cpu":
        t = t.to(device)
    if grad:
        t.requires_grad_(True)
    return t


def _cpu_list(t):
    """Flat value list on the host (Tensor.tolist is flat by contract)."""
    if t.device.type != ortho.DeviceType.CPU:
        t = t.to("cpu")
    return t.tolist()


# Storage version counter contract
class TestVersionCounterContract:
    @pytest.mark.parametrize("device", DEVICES)
    def test_new_tensor_starts_at_version_zero(self, device):
        assert _t([1.0, 2.0], device)._storage_version == 0
        assert ortho.zeros([2, 3])._storage_version == 0
        assert ortho.full([2], 7.0)._storage_version == 0

    @pytest.mark.parametrize("device", DEVICES)
    def test_views_share_the_counter_and_creation_does_not_bump(self, device):
        x = _t([[1.0, 2.0], [3.0, 4.0]], device)
        views = [x.transpose(0, 1), x.reshape([4]), x.slice(0, 0, 1), x[1],
                 x.slice(1, 0, 2)[0]]
        assert x._storage_version == 0
        for v in views:
            assert v._storage_version == 0
        x.mul_(2.0)
        # One mutation of the base is one generation observed by every alias.
        assert x._storage_version == 1
        for v in views:
            assert v._storage_version == 1

    @pytest.mark.parametrize("device", DEVICES)
    def test_mutation_through_view_bumps_the_shared_counter(self, device):
        x = _t([1.0, 2.0, 3.0, 4.0], device)
        v = x.slice(0, 1, 3)
        v.mul_(2.0)
        assert x._storage_version == 1
        assert v._storage_version == 1

    def test_each_mutation_api_bumps_exactly_once_cpu(self):
        x = _t([[1.0, 2.0], [3.0, 4.0]])
        expected = 0
        for op in (
            lambda: ortho.fill_(x, 5.0),
            lambda: x.fill_(6.0),
            lambda: x.add_(_t([[1.0, 1.0], [1.0, 1.0]])),
            lambda: x.mul_(2.0),
            lambda: x.set_item([0, 1], 9.0),
            lambda: x.__setitem__((0, 0), 4.0),        # full index: direct write
            lambda: x.__setitem__(1, 8.0),             # partial index: routes through fill_
            lambda: x._copy_data_(_t([[9.0, 9.0], [9.0, 9.0]])),
            lambda: x._copy_flat_range_from(0, _t([7.0, 7.0])),
        ):
            op()
            expected += 1
            assert x._storage_version == expected

    def test_copy_flat_range_to_bumps_destination_not_source_cpu(self):
        src = _t([1.0, 2.0, 3.0, 4.0])
        dst = _t([0.0, 0.0])
        src._copy_flat_range_to(1, dst)
        assert dst._storage_version == 1
        assert src._storage_version == 0

    @pytest.mark.parametrize("device", DEVICES)
    def test_cuda_and_cpu_inplace_ops_bump_once(self, device):
        x = _t([1.0, 2.0], device)
        x.add_(_t([1.0, 1.0], device))
        assert x._storage_version == 1
        x.mul_(0.5)
        assert x._storage_version == 2
        x._copy_data_(_t([5.0, 6.0], device))
        assert x._storage_version == 3

    def test_metadata_only_operations_do_not_bump(self):
        x = _t([[1.0, 2.0], [3.0, 4.0]])
        x.requires_grad_(True)
        x.requires_grad_(False)
        x.requires_grad_(True)
        y = x * x
        y.backward()
        x.set_grad(_t([[0.0, 0.0], [0.0, 0.0]]))
        x.zero_grad()               # clears grad METADATA, not tensor storage
        _ = x.transpose(0, 1)
        _ = x.reshape([4])
        _ = x.slice(0, 0, 1)
        _ = x[0]
        _ = x.contiguous()          # fresh storage; base untouched
        _ = x.to("cpu")             # physical copy into fresh storage
        assert x._storage_version == 0

    def test_failed_mutations_do_not_bump(self):
        x = _t([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(RuntimeError):
            x.add_(_t([1.0, 2.0, 3.0]))                     # shape mismatch
        with pytest.raises(RuntimeError):
            x.add_(_t([[1, 2], [3, 4]], dtype=np.int64))    # dtype mismatch
        with pytest.raises(RuntimeError):
            ortho.fill_(_t([[1, 2]], dtype=np.int64), 0.0)     # fill_ is float32-only
        with pytest.raises((RuntimeError, IndexError)):
            x.set_item([5, 5], 1.0)                          # out of range
        with pytest.raises(RuntimeError):
            x._copy_data_(_t([1.0, 2.0]))                    # shape mismatch
        tr = x.transpose(0, 1)
        with pytest.raises(RuntimeError):
            tr.mul_(2.0)                                     # mul_ needs contiguous
        assert x._storage_version == 0

    @pytest.mark.parametrize("device", DEVICES)
    def test_zero_numel_successful_mutation_still_bumps(self, device):
        # Deliberate policy, identical on CPU and CUDA: the counter tracks
        # successful logical mutation OPERATIONS, not bytes written.
        x = _t([1.0, 2.0, 3.0], device)
        empty_view = x.slice(0, 1, 1)
        assert empty_view.numel == 0
        if device == "cpu":
            ortho.fill_(empty_view, 9.0)
        else:
            empty_view.mul_(2.0)
        assert x._storage_version == 1
        assert _cpu_list(x) == [1.0, 2.0, 3.0]   # nothing was written

        e1 = _t([], device)
        e2 = _t([], device)
        e1._copy_data_(e2)
        assert e1._storage_version == 1
        assert e2._storage_version == 0


# Stale detection per value-sensitive Node family
class TestValueSensitiveNodesRejectStaleStorage:
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("side", ["lhs", "rhs"])
    def test_mul_rejects_stale_operand(self, device, side):
        a = _t([1.0, 2.0], device, grad=True)
        b = _t([3.0, 4.0], device, grad=True)
        y = a * b
        (a if side == "lhs" else b).mul_(2.0)
        with pytest.raises(RuntimeError, match=STALE):
            y.backward()

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("side", ["lhs", "rhs"])
    def test_matmul_rejects_stale_operand(self, device, side):
        a = _t([[1.0, 2.0], [3.0, 4.0]], device, grad=True)
        b = _t([[5.0, 6.0], [7.0, 8.0]], device, grad=True)
        y = a @ b
        (a if side == "lhs" else b).mul_(2.0)
        with pytest.raises(RuntimeError, match=STALE):
            y.backward()

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("op", ["tanh", "sigmoid", "relu", "gelu", "exp"])
    def test_value_using_activations_reject_stale_input(self, device, op):
        x = _t([0.25, -0.5, 1.0], device, grad=True)
        y = getattr(ortho, op)(x)
        x.mul_(2.0)
        with pytest.raises(RuntimeError, match=STALE):
            y.backward()

    @pytest.mark.parametrize("op", ["sqrt", "reciprocal", "leaky_relu"])
    def test_remaining_value_using_activations_reject_stale_input_cpu(self, op):
        x = _t([0.25, 0.5, 1.0], grad=True)
        y = getattr(ortho, op)(x)
        x.mul_(2.0)
        with pytest.raises(RuntimeError, match=STALE):
            y.backward()

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("op", ["softmax", "log_softmax"])
    def test_softmax_family_rejects_stale_input(self, device, op):
        x = _t([[0.5, 1.5, -0.5]], device, grad=True)
        y = getattr(ortho, op)(x, -1)
        x.mul_(2.0)
        with pytest.raises(RuntimeError, match=STALE):
            y.backward()

    @pytest.mark.parametrize("device", DEVICES)
    def test_cross_entropy_rejects_stale_logits(self, device):
        logits = _t([[1.0, 2.0, 0.5], [0.1, -0.4, 1.2]], device, grad=True)
        loss = ortho.cross_entropy(logits, [2, 0])
        logits.mul_(2.0)
        with pytest.raises(RuntimeError, match=STALE):
            loss.backward()

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("target", ["input", "weight"])
    def test_conv2d_rejects_stale_input_and_weight(self, device, target):
        x = _t(np.arange(16.0).reshape(1, 1, 4, 4), device, grad=True)
        w = _t(np.ones((1, 1, 3, 3)), device, grad=True)
        b = _t([0.5], device, grad=True)
        y = ortho.conv2d(x, w, b)
        (x if target == "input" else w).mul_(2.0)
        with pytest.raises(RuntimeError, match=STALE):
            y.backward()

    @pytest.mark.parametrize("device", DEVICES)
    def test_maxpool2d_rejects_stale_input(self, device):
        x = _t(np.arange(16.0).reshape(1, 1, 4, 4), device, grad=True)
        y = ortho.maxpool2d(x, 2, 2, 2, 2)
        x.mul_(2.0)     # backward recomputes the window argmax from values
        with pytest.raises(RuntimeError, match=STALE):
            y.backward()

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("target", ["x", "weight", "bias"])
    def test_fused_linear_relu_rejects_all_three_stale_operands(self, device, target):
        # Unlike Conv2d, the fused backward recomputes add(matmul(x, w), bias)
        # for the ReLU mask, so even the BIAS payload is value-sensitive.
        x = _t([[1.0, -2.0], [0.5, 3.0]], device, grad=True)
        w = _t([[0.5, 1.0], [-1.0, 2.0]], device, grad=True)
        b = _t([0.1, -0.2], device, grad=True)
        y = ortho.fused_linear_relu(x, w, b)
        {"x": x, "weight": w, "bias": b}[target].mul_(2.0)
        with pytest.raises(RuntimeError, match=STALE):
            y.backward()

    @pytest.mark.parametrize("device", DEVICES)
    def test_embedding_rejects_stale_indices(self, device):
        # indices_ is saved OUTSIDE Node::inputs; this is the proof that
        # version tracking follows saved VALUES, not the graph topology.
        idx = _t([0, 2, 1], device="cpu", dtype=np.int64)
        if device != "cpu":
            idx = idx.to(device)
        w = _t(np.ones((4, 3)), device, grad=True)
        y = ortho.embedding(idx, w)
        replacement = _t([1, 1, 1], device="cpu", dtype=np.int64)
        if device != "cpu":
            replacement = replacement.to(device)
        idx._copy_data_(replacement)
        with pytest.raises(RuntimeError, match=STALE):
            y.backward()

    @pytest.mark.parametrize("device", DEVICES)
    def test_nll_loss_rejects_stale_targets(self, device):
        log_probs = _t([[-1.0, -2.0, -0.5], [-0.3, -1.5, -2.0]], device, grad=True)
        targets = _t([2, 0], device="cpu", dtype=np.int64)
        if device != "cpu":
            targets = targets.to(device)
        loss = ortho.nll_loss(log_probs, targets)
        replacement = _t([0, 1], device="cpu", dtype=np.int64)
        if device != "cpu":
            replacement = replacement.to(device)
        targets._copy_data_(replacement)
        with pytest.raises(RuntimeError, match=STALE):
            loss.backward()

    def test_stale_error_names_op_role_and_versions(self):
        a = _t([1.0], grad=True)
        b = _t([2.0], grad=True)
        y = a * b
        b.mul_(2.0)
        with pytest.raises(RuntimeError,
                           match=r"MulNode as 'rhs'.*expected version 0, found 1"):
            y.backward()


# Selectivity: mathematically safe mutations stay legal
class TestValueInsensitiveNodesAllowMutation:
    @pytest.mark.parametrize("device", DEVICES)
    def test_add_backward_survives_operand_mutation(self, device):
        x = _t([1.0, 2.0], device, grad=True)
        y = _t([3.0, 4.0], device, grad=True)
        z = x + y
        x.mul_(5.0)
        y.mul_(7.0)
        z.backward()
        assert _cpu_list(x.grad) == [1.0, 1.0]
        assert _cpu_list(y.grad) == [1.0, 1.0]

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("reduce_op", ["sum", "mean"])
    def test_reductions_survive_input_mutation(self, device, reduce_op):
        x = _t([1.0, 2.0, 3.0, 4.0], device, grad=True)
        y = getattr(x, reduce_op)()
        x.mul_(2.0)
        y.backward()
        expect = 1.0 if reduce_op == "sum" else 0.25
        assert _cpu_list(x.grad) == pytest.approx([expect] * 4)

    @pytest.mark.parametrize("device", DEVICES)
    def test_pure_view_and_reshape_chain_survives_input_mutation(self, device):
        x = _t([[1.0, 2.0], [3.0, 4.0]], device, grad=True)
        y = x.transpose(0, 1).contiguous().reshape([4]).slice(0, 0, 4).sum()
        x.mul_(2.0)
        y.backward()
        assert x.grad.shape == (2, 2)
        assert _cpu_list(x.grad) == [1.0, 1.0, 1.0, 1.0]

    @pytest.mark.parametrize("device", DEVICES)
    def test_integer_index_survives_input_mutation(self, device):
        x = _t([[1.0, 2.0], [3.0, 4.0]], device, grad=True)
        y = x[1].sum()
        x.mul_(2.0)
        y.backward()
        assert x.grad.shape == (2, 2)
        assert _cpu_list(x.grad) == [0.0, 0.0, 1.0, 1.0]

    @pytest.mark.parametrize("device", DEVICES)
    def test_neg_survives_input_mutation(self, device):
        # Neg is the one ActivationKind whose backward never reads the input
        # payload (-grad on CPU, cuda_neg(grad) on CUDA).
        x = _t([1.0, -2.0], device, grad=True)
        y = -x
        x.mul_(2.0)
        y.backward()
        assert _cpu_list(x.grad) == [-1.0, -1.0]

    @pytest.mark.parametrize("device", DEVICES)
    def test_max_scatters_by_saved_argmax_after_input_mutation(self, device):
        # MaxNode saved argmax_indices_ at forward. Mutating the input is safe
        # AND the scatter must follow the ORIGINAL argmax, even though the
        # mutated values would put the max elsewhere.
        x = _t([1.0, 5.0, 3.0], device, grad=True)
        m = x.max()
        x._copy_data_(_t([100.0, 0.0, 0.0], device))   # new argmax would be 0
        m.backward()
        assert _cpu_list(x.grad) == [0.0, 1.0, 0.0]    # original argmax: 1

    @pytest.mark.parametrize("device", DEVICES)
    def test_max_dim_reduction_survives_input_mutation(self, device):
        x = _t([[1.0, 4.0], [6.0, 2.0]], device, grad=True)
        m = x.max(dim=1)
        x.mul_(2.0)
        m.backward()
        assert x.grad.shape == (2, 2)
        assert _cpu_list(x.grad) == [0.0, 1.0, 1.0, 0.0]

    @pytest.mark.parametrize("device", DEVICES)
    def test_conv2d_bias_mutation_stays_legal(self, device):
        # Verified against both backends: grad_bias is a reduction of
        # grad_output alone, so the bias payload is not watched.
        x = _t(np.arange(16.0).reshape(1, 1, 4, 4), device, grad=True)
        w = _t(np.ones((1, 1, 3, 3)), device, grad=True)
        b = _t([0.5], device, grad=True)
        y = ortho.conv2d(x, w, b)
        b.mul_(100.0)
        y.backward()
        assert _cpu_list(b.grad) == [4.0]   # 2x2 output of ones

    @pytest.mark.parametrize("device", DEVICES)
    def test_embedding_weight_mutation_stays_legal(self, device):
        # grad_weight scatters grad_output rows by the saved indices; the
        # weight payload itself is never read back.
        idx = _t([0, 2], device="cpu", dtype=np.int64)
        if device != "cpu":
            idx = idx.to(device)
        w = _t(np.ones((4, 3)), device, grad=True)
        y = ortho.embedding(idx, w)
        w.mul_(2.0)
        y.backward()
        assert _cpu_list(w.grad) == [1.0, 1.0, 1.0,
                                     0.0, 0.0, 0.0,
                                     1.0, 1.0, 1.0,
                                     0.0, 0.0, 0.0]

    @pytest.mark.parametrize("device", DEVICES)
    def test_nll_loss_log_probs_mutation_stays_legal(self, device):
        # nll_loss backward reads log_probs only for shape/dtype/device; the
        # loss VALUE depended on them, but the GRADIENT does not.
        log_probs = _t([[-1.0, -2.0, -0.5], [-0.3, -1.5, -2.0]], device, grad=True)
        loss = ortho.nll_loss(log_probs, [2, 0])
        log_probs.mul_(2.0)
        loss.backward()
        assert _cpu_list(log_probs.grad) == pytest.approx(
            [0.0, 0.0, -0.5, -0.5, 0.0, 0.0])


# Alias propagation: every view family, both directions
class TestAliasMutationPropagation:
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("family", ["slice", "index", "reshape", "chained"])
    def test_mutation_through_contiguous_view_families_rejects(self, device, family):
        x = _t([[1.0, 2.0], [3.0, 4.0]], device, grad=True)
        y = x * x
        view = {
            "slice":   lambda: x.slice(0, 0, 1),
            "index":   lambda: x[1],
            "reshape": lambda: x.reshape([4]),
            "chained": lambda: x.slice(0, 0, 2)[0],
        }[family]()
        view.mul_(2.0)
        with pytest.raises(RuntimeError, match=STALE):
            y.backward()

    def test_mutation_through_transpose_view_rejects_cpu(self):
        # A transpose view is non-contiguous, so fill_ is the mutation API
        # (add_/mul_ demand contiguity); fill_ is CPU-only.
        x = _t([[1.0, 2.0], [3.0, 4.0]], grad=True)
        y = ortho.tanh(x)
        ortho.fill_(x.transpose(0, 1), 9.0)
        with pytest.raises(RuntimeError, match=STALE):
            y.backward()

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("family", ["slice", "index", "reshape", "transpose"])
    def test_graph_on_view_rejects_base_mutation(self, device, family):
        if device == "cuda" and family == "transpose":
            # Pre-existing forward limitation, not a versioning gap: cuda_mul
            # rejects non-contiguous operands, so a graph cannot be built ON a
            # transpose view on CUDA in the first place. The transpose alias
            # direction is still covered CPU-side here and base->transpose-view
            # detection is covered on both devices above.
            pytest.skip("cuda_mul rejects non-contiguous operands (pre-existing)")
        x = _t([[1.0, 2.0], [3.0, 4.0]], device, grad=True)
        view = {
            "slice":     lambda: x.slice(0, 0, 1),
            "index":     lambda: x[0],
            "reshape":   lambda: x.reshape([4]),
            "transpose": lambda: x.transpose(0, 1),
        }[family]()
        y = view * view
        x.mul_(2.0)
        with pytest.raises(RuntimeError, match=STALE):
            y.backward()

    @pytest.mark.parametrize("device", DEVICES)
    def test_sibling_view_mutation_rejects(self, device):
        # Mutating view A must invalidate a graph built on sibling view B.
        x = _t([1.0, 2.0, 3.0, 4.0], device, grad=True)
        a = x.slice(0, 0, 2)
        b = x.slice(0, 1, 4)
        y = b * b
        a.mul_(2.0)
        with pytest.raises(RuntimeError, match=STALE):
            y.backward()


# Central validation ordering and failed-mutation hygiene
class TestBackwardValidationOrdering:
    @pytest.mark.parametrize("device", DEVICES)
    def test_preflight_fires_before_any_leaf_grad_or_hook(self, device):
        # w feeds the graph only through value-INSENSITIVE AddNode, yet the
        # stale MulNode deeper in the graph must abort the whole call before
        # w's grad is written or its hook runs.
        x = _t([1.0, 2.0], device, grad=True)
        y = _t([3.0, 4.0], device, grad=True)
        w = _t([5.0, 6.0], device, grad=True)
        fired = []
        w._register_post_accumulate_grad_hook(lambda: fired.append(True))
        z = (x * y) + w
        x.mul_(2.0)
        with pytest.raises(RuntimeError, match=STALE):
            z.backward()
        assert w.grad is None
        assert x.grad is None
        assert y.grad is None
        assert fired == []

    @pytest.mark.parametrize("device", DEVICES)
    def test_failed_backward_leaves_prior_grads_untouched(self, device):
        x = _t([1.0, 2.0], device, grad=True)
        y1 = x * x
        y1.backward()
        before = _cpu_list(x.grad)
        y2 = x * x
        x.mul_(2.0)
        with pytest.raises(RuntimeError, match=STALE):
            y2.backward()
        assert _cpu_list(x.grad) == before

    @pytest.mark.parametrize("device", DEVICES)
    def test_failed_mutation_does_not_poison_the_graph(self, device):
        x = _t([1.0, 2.0], device, grad=True)
        y = x * x
        with pytest.raises(RuntimeError):
            x.add_(_t([1.0, 2.0, 3.0], device))     # shape mismatch: no write
        with pytest.raises(RuntimeError):
            x.add_(_t([[1, 2]], dtype=np.int64) if device == "cpu"
                   else _t([1.0], device))          # dtype/shape mismatch: no write
        y.backward()
        assert _cpu_list(x.grad) == [2.0, 4.0]

    @pytest.mark.parametrize("device", DEVICES)
    def test_retained_graph_rejects_after_intervening_mutation(self, device):
        x = _t([1.0, 2.0], device, grad=True)
        y = x * x
        y.backward(retain_graph=True)
        assert _cpu_list(x.grad) == [2.0, 4.0]
        x.mul_(2.0)
        with pytest.raises(RuntimeError, match=STALE):
            y.backward()

    @pytest.mark.parametrize("device", DEVICES)
    def test_retained_graph_without_mutation_still_reruns(self, device):
        x = _t([1.0, 2.0], device, grad=True)
        y = x * x
        y.backward(retain_graph=True)
        y.backward()
        assert _cpu_list(x.grad) == [4.0, 8.0]


# Version mismatch means "mutated AFTER the save", never "ever mutated"
class TestMutationBeforeForwardStaysLegal:
    @pytest.mark.parametrize("device", DEVICES)
    def test_mutate_then_build_graph_then_backward(self, device):
        x = _t([1.0, 2.0], device, grad=True)
        x.mul_(2.0)                    # BEFORE forward: fully legal
        y = x * x
        y.backward()
        assert _cpu_list(x.grad) == [4.0, 8.0]     # 2 * (2x)

    @pytest.mark.parametrize("device", DEVICES)
    def test_mutate_between_two_independent_graphs(self, device):
        x = _t([1.0, 2.0], device, grad=True)
        y1 = x * x
        y1.backward()
        x.mul_(2.0)
        x.zero_grad()
        y2 = x * x                     # new graph captures the new version
        y2.backward()
        assert _cpu_list(x.grad) == [4.0, 8.0]

    @pytest.mark.parametrize("device", DEVICES)
    def test_non_grad_constant_is_watched_when_consumed_by_value(self, device):
        # Watching is keyed on "does backward need this value?", never on
        # requires_grad.
        x = _t([1.0, 2.0], device, grad=True)
        c = _t([3.0, 4.0], device)          # requires_grad == False
        y = x * c
        c.mul_(2.0)
        with pytest.raises(RuntimeError, match=STALE):
            y.backward()

    @pytest.mark.parametrize("device", DEVICES)
    def test_non_grad_constant_mutation_before_forward_is_legal(self, device):
        x = _t([1.0, 2.0], device, grad=True)
        c = _t([3.0, 4.0], device)
        c.mul_(2.0)
        y = x * c
        y.backward()
        assert _cpu_list(x.grad) == [6.0, 8.0]


# Optimizer and load_state_dict interaction
def _linear_model_and_loss(device):
    import orthon.nn as nn
    model = nn.Linear(2, 2)
    if device != "cpu":
        model.to("cuda")
    x = _t([[1.0, 2.0], [3.0, 4.0]], device)
    out = model(x)
    return model, (out * out).sum()


class TestOptimizerInteraction:
    @pytest.mark.parametrize("device", DEVICES)
    def test_normal_backward_then_step_stays_green(self, device):
        from orthon.optim import SGD
        model, loss = _linear_model_and_loss(device)
        opt = SGD(model.parameters(), lr=0.01)
        loss.backward()
        opt.step()
        model2, loss2 = _linear_model_and_loss(device)   # fresh forward after step
        loss2.backward()

    @pytest.mark.parametrize("device", DEVICES)
    def test_step_before_backward_invalidates_the_graph(self, device):
        from orthon.optim import SGD
        model, loss = _linear_model_and_loss(device)
        opt = SGD(model.parameters(), lr=0.01)
        loss.backward()
        opt.step()
        # Second forward captures the stepped weights; stepping AGAIN before
        # its backward mutates weight storage the saved MatMulNode depends on.
        _, loss2 = _linear_model_and_loss(device)
        out2 = model(_t([[1.0, 0.0]], device))
        loss2b = (out2 * out2).sum()
        opt.step()
        with pytest.raises(RuntimeError, match=STALE):
            loss2b.backward()

    @pytest.mark.parametrize("device", DEVICES)
    def test_retain_graph_then_step_then_second_backward_rejects(self, device):
        from orthon.optim import SGD
        model, loss = _linear_model_and_loss(device)
        opt = SGD(model.parameters(), lr=0.01)
        loss.backward(retain_graph=True)
        opt.step()
        with pytest.raises(RuntimeError, match=STALE):
            loss.backward()


class TestLoadStateDictInteraction:
    @pytest.mark.parametrize("device", DEVICES)
    def test_load_after_forward_invalidates_the_graph(self, device):
        import orthon.nn as nn
        model, loss = _linear_model_and_loss(device)
        donor = nn.Linear(2, 2)
        if device != "cpu":
            donor.to("cuda")
        weight_before = model.weight
        model.load_state_dict(donor.state_dict())
        assert model.weight is weight_before     # identity preserved
        with pytest.raises(RuntimeError, match=STALE):
            loss.backward()

    @pytest.mark.parametrize("device", DEVICES)
    def test_load_before_forward_stays_green(self, device):
        import orthon.nn as nn
        model = nn.Linear(2, 2)
        donor = nn.Linear(2, 2)
        if device != "cpu":
            model.to("cuda")
            donor.to("cuda")
        model.load_state_dict(donor.state_dict())
        x = _t([[1.0, 2.0]], device)
        out = model(x)
        (out * out).sum().backward()
        assert model.weight.grad is not None


# AMP mutation coverage
class TestAmpCopyMutation:
    @requires_cuda
    def test_amp_master_to_model_copy_bumps_once_and_invalidates(self):
        import orthon._C as _C
        model_p = _t([1.0, 2.0], "cuda", grad=True)
        master = _t([5.0, 6.0], "cuda")
        y = model_p * model_p
        _C._amp_copy_master_to_model_(model_p, master)
        assert model_p._storage_version == 1     # exactly one bump
        with pytest.raises(RuntimeError, match=STALE):
            y.backward()

    @requires_cuda
    def test_amp_unscale_helper_does_not_mutate_its_input(self):
        import orthon._C as _C
        grad = _t([2.0, 4.0], "cuda")
        _C._amp_unscale_to_float32_and_check(grad, 0.5)
        assert grad._storage_version == 0
        assert _cpu_list(grad) == [2.0, 4.0]


# Distributed (local backend) mutation coverage (single-process)
class TestLocalDistributedMutation:
    def test_local_broadcast_copy_bumps_destination(self):
        # The local backend's broadcast copies via _copy_flat_range_from, so
        # the destination inherits the bump from the one shared copy helper.
        dst = _t([0.0, 0.0, 0.0])
        src = _t([1.0, 2.0, 3.0])
        dst._copy_flat_range_from(0, src)
        assert dst._storage_version == 1
        assert src._storage_version == 0
        assert _cpu_list(dst) == [1.0, 2.0, 3.0]
