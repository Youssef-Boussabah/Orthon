import orthon as ortho
from .module import Module, Parameter


def _reduce_mean(tensor, dims):
    """Mean over several dimensions using the single-dim primitive.

    keepdim=True preserves rank, so the remaining dimension indices stay valid
    across successive reductions.
    """
    out = tensor
    for dim in dims:
        out = out.mean(dim=dim, keepdim=True)
    return out


class _BatchNorm(Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ):
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.affine = bool(affine)
        if affine:
            self.weight = Parameter(ortho.ones([num_features]))
            self.bias = Parameter(ortho.zeros([num_features]))
        # Persistent, non-trainable model state. Registered as buffers so
        # they take part in state_dict/load_state_dict/.to()/.cast().
        self.register_buffer("running_mean", ortho.zeros([self.num_features]))
        self.register_buffer("running_var", ortho.ones([self.num_features]))

    def _axes(self, ndim):
        raise NotImplementedError

    def _stat_shape(self, ndim):
        raise NotImplementedError

    def _check_input(self, x):
        if x.shape[1] != self.num_features:
            raise ValueError(
                f"expected {self.num_features} channels, got {x.shape[1]}"
            )

    def _value_count(self, x):
        """Number of values reduced per channel (PyTorch's M)."""
        count = 1
        for dim in self._axes(x.ndim):
            count *= int(x.shape[dim])
        return count

    def _detached_stat(self, tensor, buffer):
        """Copy a statistic to independent buffer-shaped state under no_grad.

        Same-device to() still copies; explicit detachment prevents a batch graph
        from surviving in running statistics."""
        with ortho.no_grad():
            out = tensor.reshape([self.num_features]).to(buffer.device)
            if out.dtype != buffer.dtype:
                out = ortho.cast(out, buffer.dtype)
        return out

    def _update_running_stats(self, mean, biased_var, count):
        """running = (1 - momentum) * running + momentum * batch.

        Matches PyTorch: the running variance tracks the *unbiased* batch
        variance (denominator M - 1) while normalisation uses the biased one.
        """
        # The correction factor, the momentum and its complement are Python
        # scalars and are written as such, rather than materialised as
        # [num_features] constant tensors. The statistics are detached first
        # by _detached_stat, which is an explicit no_grad snapshot.
        batch_mean = self._detached_stat(mean, self.running_mean)
        batch_var = self._detached_stat(biased_var, self.running_var)

        unbiased_var = batch_var * (count / (count - 1))

        keep = 1.0 - self.momentum
        self.running_mean = self.running_mean * keep + batch_mean * self.momentum
        self.running_var = self.running_var * keep + unbiased_var * self.momentum

    def forward(self, x):
        self._check_input(x)
        stat_shape = list(self._stat_shape(x.ndim))

        if self.training:
            count = self._value_count(x)
            if count < 2:
                raise ValueError(
                    "Expected more than 1 value per channel when training, "
                    f"got input size {list(x.shape)}"
                )
            dims = self._axes(x.ndim)
            mean = _reduce_mean(x, dims)
            centered = x - mean
            # Biased (population) variance is what normalisation uses.
            var = _reduce_mean(centered * centered, dims)
            self._update_running_stats(mean, var, count)
        else:
            mean = self.running_mean.reshape(stat_shape)
            var = self.running_var.reshape(stat_shape)
            centered = x - mean

        inv_std = ortho.reciprocal(ortho.sqrt(var + self.eps))
        y = centered * inv_std

        if self.affine:
            y = y * self.weight.reshape(stat_shape)
            y = y + self.bias.reshape(stat_shape)
        return y


class BatchNorm1d(_BatchNorm):
    """Minimal BatchNorm1d for 2D inputs shaped [N, C]."""

    def _axes(self, ndim):
        if ndim != 2:
            raise ValueError("BatchNorm1d expects a 2D [N, C] tensor")
        return (0,)

    def _stat_shape(self, ndim):
        if ndim != 2:
            raise ValueError("BatchNorm1d expects a 2D [N, C] tensor")
        return (1, self.num_features)


class BatchNorm2d(_BatchNorm):
    """Minimal BatchNorm2d for 4D inputs shaped [N, C, H, W]."""

    def _axes(self, ndim):
        if ndim != 4:
            raise ValueError("BatchNorm2d expects a 4D [N, C, H, W] tensor")
        return (0, 2, 3)

    def _stat_shape(self, ndim):
        if ndim != 4:
            raise ValueError("BatchNorm2d expects a 4D [N, C, H, W] tensor")
        return (1, self.num_features, 1, 1)
