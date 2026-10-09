"""Shared small token model for nn, optimizer and AMP specifications."""
from orthon import nn


class TokenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(7, 4)
        self.block = nn.TransformerEncoderBlock(4, 2, mlp_ratio=2)
        self.head = nn.Linear(4, 2)

    def forward(self, indices):
        return self.head(self.block(self.embedding(indices)).mean(dim=1))
