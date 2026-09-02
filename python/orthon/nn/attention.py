import orthon as ortho
from orthon import functional as F
from .linear import Linear
from .module import Module


def _linear_3d(layer, x):
    b, t, d = [int(v) for v in x.shape]
    y = layer(x.reshape([b * t, d]))
    return y.reshape([b, t, int(y.shape[-1])])


class SelfAttention(Module):
    """Single-head self-attention over [B, T, E] tensors."""

    def __init__(self, embed_dim, dropout=0.0):
        super().__init__()
        self.embed_dim = int(embed_dim)
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        self.dropout = float(dropout)
        self.q_proj = Linear(self.embed_dim, self.embed_dim)
        self.k_proj = Linear(self.embed_dim, self.embed_dim)
        self.v_proj = Linear(self.embed_dim, self.embed_dim)
        self.out_proj = Linear(self.embed_dim, self.embed_dim)

    def forward(self, x, mask=None):
        if x.ndim != 3:
            raise ValueError("SelfAttention expects input with shape [B, T, E]")
        if int(x.shape[-1]) != self.embed_dim:
            raise ValueError("SelfAttention input embedding dimension mismatch")
        q = _linear_3d(self.q_proj, x)
        k = _linear_3d(self.k_proj, x)
        v = _linear_3d(self.v_proj, x)
        y = ortho.scaled_dot_product_attention(
            q,
            k,
            v,
            mask=mask,
            dropout_p=self.dropout,
            training=self.training,
        )
        return _linear_3d(self.out_proj, y)


class MultiHeadSelfAttention(Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()

        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

    def _project(self, proj, x, B, T, E):
        x_flat = x.reshape([B * T, E])
        y_flat = proj(x_flat)
        return y_flat.reshape([B, T, E])

    def forward(self, x, mask=None):
        if len(x.shape) != 3:
            raise ValueError("MultiHeadSelfAttention expects input shape [B, T, E]")

        B, T, E = x.shape
        if E != self.embed_dim:
            raise ValueError(f"expected embed_dim={self.embed_dim}, got {E}")

        q = self._project(self.q_proj, x, B, T, E)
        k = self._project(self.k_proj, x, B, T, E)
        v = self._project(self.v_proj, x, B, T, E)

        q = q.reshape([B, T, self.num_heads, self.head_dim]).transpose(1, 2).contiguous()
        k = k.reshape([B, T, self.num_heads, self.head_dim]).transpose(1, 2).contiguous()
        v = v.reshape([B, T, self.num_heads, self.head_dim]).transpose(1, 2).contiguous()

        q = q.reshape([B * self.num_heads, T, self.head_dim])
        k = k.reshape([B * self.num_heads, T, self.head_dim])
        v = v.reshape([B * self.num_heads, T, self.head_dim])

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            mask=mask,
            dropout_p=self.dropout,
            training=self.training,
        )

        attn = attn.reshape([B, self.num_heads, T, self.head_dim])
        attn = attn.transpose(1, 2).contiguous()
        attn = attn.reshape([B, T, E])

        out_flat = self.out_proj(attn.reshape([B * T, E]))
        return out_flat.reshape([B, T, E])
