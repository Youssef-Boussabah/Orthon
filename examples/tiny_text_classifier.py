import argparse
import random

import numpy as np

import orthon as ortho
import orthon.nn as nn


def _indices(values, device):
    t = ortho.from_numpy(np.asarray(values, dtype=np.int64))
    if device == "cuda":
        t = t.to("cuda")
    return t


def make_batch(batch_size, seq_len, vocab_size, device, seed):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, vocab_size, size=(batch_size, seq_len), dtype=np.int64)
    y = np.zeros((batch_size,), dtype=np.int64)

    half = batch_size // 2
    x[:half] = rng.integers(0, vocab_size - 1, size=(half, seq_len), dtype=np.int64)
    insert_pos = rng.integers(0, seq_len, size=(half,))
    x[np.arange(half), insert_pos] = 7
    y[:half] = 1

    x[half:] = rng.integers(0, vocab_size - 1, size=(batch_size - half, seq_len), dtype=np.int64)
    x[half:][x[half:] >= 7] += 1
    order = rng.permutation(batch_size)
    return _indices(x[order], device), _indices(y[order], device)


class TinyTextClassifier(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.seq_len = seq_len
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(seq_len, embed_dim)
        self.attention = nn.SelfAttention(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, tokens):
        batch = int(tokens.shape[0])
        positions = _indices([list(range(self.seq_len)) for _ in range(batch)],
                             "cuda" if tokens.device.is_cuda() else "cpu")
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        x = self.attention(x)
        x = self.norm(x.mean(dim=1))
        x = ortho.gelu(self.fc1(x))
        return self.fc2(x)


def accuracy(logits, targets):
    values = logits.to("cpu").tolist() if logits.device.is_cuda() else logits.tolist()
    labels = targets.to("cpu").tolist() if targets.device.is_cuda() else targets.tolist()
    classes = int(logits.shape[1])
    correct = 0
    for i, label in enumerate(labels):
        row = values[i * classes:(i + 1) * classes]
        pred = max(range(classes), key=lambda c: row[c])
        correct += int(pred == label)
    return correct / len(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.05)
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    ortho.manual_seed(0)   # The framework RNG is separate from Python's global random.

    model = TinyTextClassifier(
        vocab_size=32,
        seq_len=8,
        embed_dim=12,
        hidden_dim=16,
        num_classes=2,
    )
    if args.device == "cuda":
        model = model.to("cuda")

    optimizer = ortho.optim.SGD(model.parameters(), lr=args.lr)
    first_loss = None
    last_loss = None
    last_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        tokens, targets = make_batch(args.batch_size, 8, 32, args.device, seed=epoch)
        optimizer.zero_grad()
        logits = model(tokens)
        log_probs = ortho.log_softmax(logits, dim=-1)
        loss = ortho.nll_loss(log_probs, targets)
        loss.backward()
        optimizer.step()

        last_loss = loss.item()
        if first_loss is None:
            first_loss = last_loss
        last_acc = accuracy(logits, targets)
        print(f"epoch {epoch:02d} loss={last_loss:.4f} accuracy={last_acc:.3f}")

    print(
        f"summary first_loss={first_loss:.4f} "
        f"last_loss={last_loss:.4f} last_accuracy={last_acc:.3f}"
    )


if __name__ == "__main__":
    main()
