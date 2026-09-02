"""Tests for Module.state_dict / load_state_dict and file save/load.

The file round-trips below go through the unified pickle-free archive.
Detection never looks at the extension, and the assertions are on EXACT bytes:
with a raw-byte codec there is no reason for a saved value to come back merely
close to what was written.
"""
import tempfile
import os
import pytest
import orthon
from orthon.nn import Linear, ReLU, Sequential, Conv2d


def _make_model():
    return Sequential(Linear(4, 8), ReLU(), Linear(8, 2))


def _raw(tensor):
    with orthon.no_grad():
        return orthon._tensor_to_bytes(tensor.to("cpu"))


# state_dict keys.

def test_state_dict_linear_keys():
    model = Linear(4, 2)
    sd = model.state_dict()
    assert set(sd.keys()) == {'weight', 'bias'}


def test_state_dict_sequential_keys():
    model = _make_model()
    sd = model.state_dict()
    assert set(sd.keys()) == {'0.weight', '0.bias', '2.weight', '2.bias'}


def test_state_dict_nested_keys():
    """ReLU has no parameters; its absence is correct."""
    model = Sequential(Linear(2, 4), ReLU(), Linear(4, 1))
    sd = model.state_dict()
    assert '1.weight' not in sd      # ReLU has no params
    assert '0.weight' in sd
    assert '2.weight' in sd


def test_state_dict_values_are_tensors():
    model = Linear(4, 2)
    for v in model.state_dict().values():
        assert isinstance(v, orthon.Tensor)


def test_state_dict_shapes():
    model = Linear(4, 2)
    sd = model.state_dict()
    assert list(sd['weight'].shape) == [4, 2]
    assert list(sd['bias'].shape)   == [2]


# load_state_dict.

def test_load_state_dict_restores_values():
    model = Linear(4, 2)
    # Manually set weight to known values
    for i in range(4):
        for j in range(2):
            model.weight.set_item([i, j], float(i * 2 + j + 1))

    sd = model.state_dict()

    # Build a fresh model and load
    model2 = Linear(4, 2)
    model2.load_state_dict(sd)

    assert model2.weight.tolist() == pytest.approx(model.weight.tolist())
    assert model2.bias.tolist()   == pytest.approx(model.bias.tolist())


def test_load_state_dict_preserves_requires_grad():
    model  = Linear(4, 2)
    model2 = Linear(4, 2)
    model2.load_state_dict(model.state_dict())
    for p in model2.parameters():
        assert p.requires_grad


def test_load_state_dict_shape_mismatch_raises():
    model1 = Linear(4, 2)
    model2 = Linear(4, 3)   # different out_features
    with pytest.raises((ValueError, KeyError)):
        model2.load_state_dict(model1.state_dict())


def test_load_state_dict_missing_key_raises():
    model = Linear(4, 2)
    sd = model.state_dict()
    del sd['bias']
    with pytest.raises(KeyError):
        model.load_state_dict(sd, strict=True)


def test_load_state_dict_unexpected_key_raises():
    model = Linear(4, 2)
    sd = model.state_dict()
    sd['extra'] = orthon.zeros([1])
    with pytest.raises(KeyError):
        model.load_state_dict(sd, strict=True)


def test_load_state_dict_non_strict_ignores_missing():
    model = Linear(4, 2)
    sd = {'weight': model.state_dict()['weight']}   # no bias
    # Should not raise with strict=False
    model.load_state_dict(sd, strict=False)


# File save/load round-trip.

def _tempfile():
    # The suffix is deliberately not ".pkl" any more, and deliberately not a
    # format-specific one either: the loader identifies files by content.
    with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as f:
        return f.name


def test_save_load_roundtrip_linear():
    model1 = Linear(4, 2)
    # Set distinctive values
    for i in range(4):
        for j in range(2):
            model1.weight.set_item([i, j], float(i + j) * 0.3)

    path = _tempfile()
    try:
        orthon.save_state_dict(model1, path)

        model2 = Linear(4, 2)
        orthon.load_state_dict(model2, path)

        assert _raw(model2.weight) == _raw(model1.weight)
        assert _raw(model2.bias) == _raw(model1.bias)
    finally:
        os.unlink(path)


def test_save_load_roundtrip_sequential():
    model1 = _make_model()

    path = _tempfile()
    try:
        orthon.save_state_dict(model1, path)

        model2 = _make_model()
        orthon.load_state_dict(model2, path)

        sd1 = model1.state_dict()
        sd2 = model2.state_dict()
        assert sd1.keys() == sd2.keys()
        for key in sd1:
            assert _raw(sd2[key]) == _raw(sd1[key]), key
    finally:
        os.unlink(path)


def test_save_load_roundtrip_conv2d():
    model1 = Sequential(
        Conv2d(1, 4, kernel_size=3, padding=1),
        ReLU(),
        Linear(4, 2),
    )
    path = _tempfile()
    try:
        orthon.save_state_dict(model1, path)

        model2 = Sequential(
            Conv2d(1, 4, kernel_size=3, padding=1),
            ReLU(),
            Linear(4, 2),
        )
        orthon.load_state_dict(model2, path)

        sd1, sd2 = model1.state_dict(), model2.state_dict()
        assert sd1.keys() == sd2.keys()
        for key in sd1:
            assert _raw(sd2[key]) == _raw(sd1[key]), key
    finally:
        os.unlink(path)


def test_save_load_roundtrip_preserves_parameter_identity():
    """The file APIs copy INTO parameters; they never replace them."""
    model1 = _make_model()
    path = _tempfile()
    try:
        orthon.save_state_dict(model1, path)
        model2 = _make_model()
        before = [p._impl_id for p in model2.parameters()]
        orthon.load_state_dict(model2, path)
        assert [p._impl_id for p in model2.parameters()] == before
    finally:
        os.unlink(path)
