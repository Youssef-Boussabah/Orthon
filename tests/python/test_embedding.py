from tests.support.capabilities import orthon_cuda_available as _cuda_available
import numpy as np
import pytest

import orthon as ortho
import orthon.functional as F
import orthon.nn as nn

torch = pytest.importorskip("torch")




requires_cuda = pytest.mark.skipif(not (torch.cuda.is_available() and _cuda_available()), reason="CUDA not available")


def _indices(values, device="cpu", dtype=np.int64):
    t = ortho.from_numpy(np.asarray(values, dtype=dtype))
    if device == "cuda":
        t = t.to("cuda")
    return t


def _fill_weight(weight, num_embeddings, embedding_dim):
    for i in range(num_embeddings):
        for j in range(embedding_dim):
            weight.set_item([i, j], float(i * 10 + j + 1) / 10.0)


def _new_embedding(num_embeddings=5, embedding_dim=3, padding_idx=None, device="cpu"):
    layer = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    _fill_weight(layer.weight, num_embeddings, embedding_dim)
    if padding_idx is not None:
        for j in range(embedding_dim):
            layer.weight.set_item([padding_idx, j], 0.0)
    if device == "cuda":
        layer = layer.to("cuda")
    return layer


def _torch_embedding_from_orthon(layer, padding_idx=None, device="cpu"):
    weight = torch.tensor(
        layer.weight.to("cpu").tolist(),
        dtype=torch.float32,
        device=device,
    ).reshape(layer.num_embeddings, layer.embedding_dim)
    emb = torch.nn.Embedding(
        layer.num_embeddings,
        layer.embedding_dim,
        padding_idx=padding_idx,
        device=device,
    )
    with torch.no_grad():
        emb.weight.copy_(weight)
    return emb


def _assert_close(actual, expected, rel=1e-5, abs=1e-5):
    if actual.device.is_cuda():
        actual = actual.to("cpu")
    assert list(actual.shape) == list(expected.shape)
    assert actual.tolist() == pytest.approx(
        expected.detach().cpu().flatten().tolist(), rel=rel, abs=abs
    )


@pytest.mark.parametrize("ids", [[0, 2, 4], [[0, 1, 2], [3, 2, 1]]])
def test_embedding_cpu_forward_matches_torch(ids):
    layer = _new_embedding()
    y = layer(_indices(ids))

    torch_layer = _torch_embedding_from_orthon(layer)
    ty = torch_layer(torch.tensor(ids, dtype=torch.long))

    _assert_close(y, ty)


def test_functional_embedding_accepts_python_lists():
    layer = _new_embedding()
    y = F.embedding([[1, 2], [3, 4]], layer.weight)
    assert list(y.shape) == [2, 2, 3]


def test_embedding_cpu_backward_repeated_indices_match_torch():
    ids = [1, 1, 3, 1]
    layer = _new_embedding()
    y = layer(_indices(ids))
    y.sum().backward()

    torch_layer = _torch_embedding_from_orthon(layer)
    ty = torch_layer(torch.tensor(ids, dtype=torch.long))
    ty.sum().backward()

    assert layer.weight.grad.tolist() == pytest.approx(
        torch_layer.weight.grad.detach().flatten().tolist()
    )


def test_embedding_cpu_padding_idx_has_zero_row_and_no_grad():
    ids = [[0, 1, 0], [2, 0, 3]]
    layer = nn.Embedding(4, 2, padding_idx=0)
    assert layer.weight[0].tolist() == pytest.approx([0.0, 0.0])
    y = layer(_indices(ids))
    y.sum().backward()

    assert layer.weight.grad.tolist()[0:2] == pytest.approx([0.0, 0.0])


def test_embedding_module_parameters():
    layer = nn.Embedding(6, 4)
    params = list(layer.parameters())
    assert params == [layer.weight]
    assert list(layer.weight.shape) == [6, 4]


def test_embedding_invalid_cpu_index_raises():
    layer = nn.Embedding(3, 2)
    with pytest.raises((RuntimeError, IndexError, ValueError)):
        layer(_indices([0, 3]))


def test_embedding_simple_cpu_model_backward():
    embed = nn.Embedding(8, 4)
    linear = nn.Linear(4, 3)
    ids = _indices([[1, 2, 3], [2, 3, 4]])
    logits = linear(embed(ids).mean(dim=1))
    loss = ortho.cross_entropy(logits, [0, 2])
    loss.backward()
    assert embed.weight.grad is not None
    assert linear.weight.grad is not None


@requires_cuda
@pytest.mark.parametrize("ids", [[0, 2, 4], [[0, 1, 2], [3, 2, 1]]])
def test_embedding_cuda_forward_matches_torch(ids):
    layer = _new_embedding(device="cuda")
    y = layer(_indices(ids, device="cuda"))

    torch_layer = _torch_embedding_from_orthon(layer, device="cuda")
    ty = torch_layer(torch.tensor(ids, dtype=torch.long, device="cuda"))

    assert y.device.is_cuda()
    _assert_close(y, ty)


@requires_cuda
def test_embedding_cuda_backward_repeated_indices_match_torch():
    ids = [[1, 1, 3], [4, 1, 3]]
    layer = _new_embedding(device="cuda")
    y = layer(_indices(ids, device="cuda"))
    y.sum().backward()

    torch_layer = _torch_embedding_from_orthon(layer, device="cuda")
    ty = torch_layer(torch.tensor(ids, dtype=torch.long, device="cuda"))
    ty.sum().backward()

    assert layer.weight.grad.device.is_cuda()
    assert layer.weight.grad.to("cpu").tolist() == pytest.approx(
        torch_layer.weight.grad.detach().cpu().flatten().tolist()
    )


@requires_cuda
def test_embedding_cuda_padding_idx_has_no_grad():
    ids = [[0, 1, 0], [2, 0, 3]]
    layer = _new_embedding(4, 2, padding_idx=0, device="cuda")
    y = layer(_indices(ids, device="cuda"))
    y.sum().backward()

    assert y.device.is_cuda()
    assert layer.weight.grad.device.is_cuda()
    assert layer.weight.grad.to("cpu").tolist()[0:2] == pytest.approx([0.0, 0.0])


@requires_cuda
def test_embedding_module_to_cuda_moves_weight():
    layer = nn.Embedding(6, 4).to("cuda")
    assert layer.weight.device.is_cuda()
    y = layer(_indices([1, 2, 3], device="cuda"))
    assert y.device.is_cuda()


@requires_cuda
def test_embedding_simple_cuda_model_backward():
    embed = nn.Embedding(8, 4).to("cuda")
    linear = nn.Linear(4, 3).to("cuda")
    ids = _indices([[1, 2, 3], [2, 3, 4]], device="cuda")
    logits = linear(embed(ids).mean(dim=1))
    loss = ortho.cross_entropy(logits, [0, 2])
    loss.backward()
    assert embed.weight.grad.device.is_cuda()
    assert linear.weight.grad.device.is_cuda()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda)])
@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_r1_embedding_endpoints_duplicates_and_negative_padding(device, dtype):
    weight = ortho.from_numpy(np.arange(8, dtype=np.float32).reshape(4, 2)).to(device)
    weight.requires_grad_(True)
    ids = _indices([0, 3, 0, 2], device, dtype)
    output = ortho.embedding(ids, weight, padding_idx=-1)
    assert output.to("cpu").tolist() == [0, 1, 6, 7, 0, 1, 4, 5]
    output.sum().backward()
    assert weight.grad.to("cpu").tolist() == [2, 2, 0, 0, 1, 1, 0, 0]


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda)])
@pytest.mark.parametrize("dtype,value", [(np.int32, -1), (np.int32, 4), (np.int32, 2**31-1),
                                        (np.int64, -1), (np.int64, 4), (np.int64, 2**32),
                                        (np.int64, 2**63-1)])
def test_r1_embedding_rejects_index_domain_before_gather(device, dtype, value):
    weight = ortho.ones([4, 2]).to(device).requires_grad_(True)
    version = weight._storage_version
    with pytest.raises(IndexError, match="out of range"):
        ortho.embedding(_indices([0, value], device, dtype), weight)
    assert weight._storage_version == version and weight.grad is None
    assert weight.to("cpu").tolist() == [1]*8


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda)])
def test_r1_embedding_empty_indices_have_zero_gradient(device):
    weight = ortho.ones([3, 2]).to(device).requires_grad_(True)
    result = ortho.embedding(_indices([], device), weight)
    assert result.shape == (0, 2)
    result.backward()
    assert weight.grad.to("cpu").tolist() == [0]*6
