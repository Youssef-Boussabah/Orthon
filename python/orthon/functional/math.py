from orthon._random import randbits as _randbits
import orthon as ortho
from orthon.nn.layernorm import layer_norm as _layer_norm
from orthon.nn.embedding import embedding as _embedding


def relu(x):
    return ortho.relu(x)


def sigmoid(x):
    return ortho.sigmoid(x)


def tanh(x):
    return ortho.tanh(x)


def leaky_relu(x, negative_slope=0.01):
    return ortho.leaky_relu(x, negative_slope)


def gelu(x):
    return ortho.gelu(x)


def exp(x):
    return ortho.exp(x)


def log_softmax(x, dim=-1):
    return ortho.log_softmax(x, dim=dim)


def nll_loss(log_probs, targets, reduction="mean"):
    return ortho.nll_loss(log_probs, targets, reduction=reduction)


def scaled_dot_product_attention(q, k, v, mask=None, dropout_p=0.0, training=True):
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("scaled_dot_product_attention expects q, k, v with shape [B, T, D]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("scaled_dot_product_attention expects q, k, v to have matching shapes")
    if q.device != k.device or q.device != v.device:
        raise ValueError("scaled_dot_product_attention expects q, k, v on the same device")
    d = int(q.shape[-1])
    if d <= 0:
        raise ValueError("scaled_dot_product_attention requires non-empty feature dimension")

    k_t = k.transpose(-2, -1)

    # The scale is a Python scalar operand, which carries the tensor's dtype
    # and exact Device (cuda:N included) by contract. A helper that built a
    # [1]-element constant tensor and placed it would have to remember to pass
    # the Device object rather than a "cuda" string, which resolves to cuda:0.
    scores = ortho.matmul(q, k_t) * (d ** -0.5)
    if mask is not None:
        scores = scores + mask

    weights = ortho.exp(ortho.log_softmax(scores, dim=-1))
    if dropout_p > 0.0 and training:
        # Widened from randrange(0, 2**63) to a full 64-bit draw: the native
        # binding takes uint64_t and nn.Dropout already passes full-width seeds,
        # so the old 63-bit range was an unnecessary restriction, not a
        # compatibility requirement. The draw stays inside this branch, so a
        # disabled or eval-mode attention call does not advance the stream.
        weights = ortho.dropout(
            weights,
            float(dropout_p),
            _randbits(64),
        )
    return ortho.matmul(weights, v)


def sum(x, dim=None, keepdim=False):
    return ortho.sum(x, dim=dim, keepdim=keepdim)


def mean(x, dim=None, keepdim=False):
    return ortho.mean(x, dim=dim, keepdim=keepdim)


def max(x, dim=None, keepdim=False):
    return ortho.max(x, dim=dim, keepdim=keepdim)


def argmax(x, dim=None, keepdim=False):
    return ortho.argmax(x, dim=dim, keepdim=keepdim)


def layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    return _layer_norm(x, normalized_shape, weight=weight, bias=bias, eps=eps)


def embedding(input, weight, padding_idx=None):
    return _embedding(input, weight, padding_idx=padding_idx)
