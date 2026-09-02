import orthon as ortho
from .module import Module, Parameter


def _normalized_shape_tuple(normalized_shape):
    if isinstance(normalized_shape, int):
        shape = (int(normalized_shape),)
    else:
        try:
            shape = tuple(int(dim) for dim in normalized_shape)
        except TypeError as exc:
            raise TypeError("normalized_shape must be an int or sequence of ints") from exc
    if not shape:
        raise ValueError("normalized_shape must contain at least one dimension")
    if any(dim <= 0 for dim in shape):
        raise ValueError("normalized_shape dimensions must be positive")
    return shape


def _check_input_shape(x, normalized_shape):
    if len(normalized_shape) > x.ndim:
        raise ValueError(
            "normalized_shape cannot be longer than input rank: "
            f"{normalized_shape} vs input shape {list(x.shape)}"
        )
    actual = tuple(int(dim) for dim in x.shape[-len(normalized_shape):])
    if actual != normalized_shape:
        raise ValueError(
            "LayerNorm expected trailing input shape "
            f"{normalized_shape}, got {actual}"
        )


def layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    normalized_shape = _normalized_shape_tuple(normalized_shape)
    _check_input_shape(x, normalized_shape)

    normalized_dims = range(x.ndim - len(normalized_shape), x.ndim)

    mean = x
    for dim in reversed(list(normalized_dims)):
        mean = mean.mean(dim=dim, keepdim=True)

    # Written with plain scalar arithmetic. Spelling this as
    # `x + _negate(mean)` would multiply by a statistic-shaped constant -1
    # tensor, and eps would be another such constant, each needing a helper to
    # place it on the caller's exact device. The scalar contract carries dtype
    # and exact Device (cuda:N included) by construction instead.
    centered = x - mean
    var = centered * centered
    for dim in reversed(list(normalized_dims)):
        var = var.mean(dim=dim, keepdim=True)

    inv_std = ortho.reciprocal(ortho.sqrt(var + eps))
    y = centered * inv_std

    if weight is not None:
        y = y * weight
    if bias is not None:
        y = y + bias
    return y


class LayerNorm(Module):
    """Layer normalization over the trailing normalized_shape dimensions."""

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, device=None):
        super().__init__()
        self.normalized_shape = _normalized_shape_tuple(normalized_shape)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        if self.elementwise_affine:
            weight = ortho.ones(list(self.normalized_shape))
            bias = ortho.zeros(list(self.normalized_shape))
            if device is not None:
                weight = weight.to(device)
                bias = bias.to(device)
            self.weight = Parameter(weight)
            self.bias = Parameter(bias)

    def forward(self, x):
        weight = self.weight if self.elementwise_affine else None
        bias = self.bias if self.elementwise_affine else None
        return layer_norm(
            x,
            self.normalized_shape,
            weight=weight,
            bias=bias,
            eps=self.eps,
        )
