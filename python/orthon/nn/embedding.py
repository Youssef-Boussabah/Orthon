import math

import numpy as np

import orthon as ortho
from .._random import uniform as _uniform
from .module import Module, Parameter


def _normalize_padding_idx(padding_idx, num_embeddings):
    if padding_idx is None:
        return None
    padding_idx = int(padding_idx)
    if padding_idx < 0:
        padding_idx += num_embeddings
    if padding_idx < 0 or padding_idx >= num_embeddings:
        raise ValueError("padding_idx must be within num_embeddings")
    return padding_idx


def _indices_tensor(input, device=None):
    """Materialise host index data onto the WEIGHT's exact device.

    ``device`` is the weight's Device, and the move must target it exactly:
    a bare ``"cuda"`` spec parses to cuda:0, so an Embedding whose weight
    lives on cuda:1 would otherwise generate its indices on cuda:0 and trip
    "embedding: indices and weight must be on the same device".
    """
    if isinstance(input, ortho.Tensor):
        return input
    arr = np.asarray(input, dtype=np.int64)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    t = ortho.from_numpy(arr)
    if device is not None and device.is_cuda():
        t = t.to(device)
    return t


def _init_weight(num_embeddings, embedding_dim, padding_idx):
    bound = 1.0 / math.sqrt(max(1, num_embeddings))
    t = ortho.empty([num_embeddings, embedding_dim])
    for i in range(num_embeddings):
        for j in range(embedding_dim):
            value = 0.0 if i == padding_idx else _uniform(-bound, bound)
            t.set_item([i, j], value)
    return t


def embedding(input, weight, padding_idx=None):
    indices = _indices_tensor(input, weight.device)
    return ortho.embedding(indices, weight, padding_idx)


class Embedding(Module):
    """Lookup table that maps integer token IDs to embedding vectors."""

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, device=None):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        if self.num_embeddings <= 0:
            raise ValueError("num_embeddings must be positive")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.padding_idx = _normalize_padding_idx(padding_idx, self.num_embeddings)

        weight = _init_weight(self.num_embeddings, self.embedding_dim, self.padding_idx)
        if device is not None:
            weight = weight.to(device)
        self.weight = Parameter(weight)

    def forward(self, input):
        return embedding(input, self.weight, self.padding_idx)
