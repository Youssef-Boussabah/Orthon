import orthon as ortho
from .module import Module
from .linear import Linear
from .layernorm import LayerNorm
from .dropout import Dropout
from .attention import MultiHeadSelfAttention


def _linear_3d(linear, x):
    b, t, d = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    out_d = int(linear.weight.shape[1])
    y = linear(x.reshape([b * t, d]))
    return y.reshape([b, t, out_d])


class TransformerEncoderBlock(Module):
    """Pre-norm Transformer encoder block: LN → MHA → Add, LN → MLP → Add.

    Forward shape: [B, T, E] → [B, T, E].
    """

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0, eps=1e-5):
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        hidden_dim = int(embed_dim * mlp_ratio)

        self.ln1 = LayerNorm(embed_dim, eps=eps)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout=dropout)
        self.dropout1 = Dropout(dropout) if dropout > 0.0 else None

        self.ln2 = LayerNorm(embed_dim, eps=eps)
        self.fc1 = Linear(embed_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, embed_dim)
        self.dropout2 = Dropout(dropout) if dropout > 0.0 else None
        self.dropout3 = Dropout(dropout) if dropout > 0.0 else None

    def forward(self, x, mask=None):
        h = self.ln1(x)
        h = self.attn(h, mask=mask)
        if self.dropout1 is not None:
            h = self.dropout1(h)
        x = x + h

        h = self.ln2(x)
        h = _linear_3d(self.fc1, h)
        h = ortho.gelu(h)
        if self.dropout2 is not None:
            h = self.dropout2(h)
        h = _linear_3d(self.fc2, h)
        if self.dropout3 is not None:
            h = self.dropout3(h)
        x = x + h

        return x
