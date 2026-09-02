"""Toy Transformer sequence-classification demo using Orthon.

Synthetic task: vocab_size=32, seq_len=8, num_classes=2.
Label = 1 if the special_token (7) appears in the sequence, else 0.
Train/test sets are balanced 50/50 and generated from a fixed NumPy seed.

Usage:
    python examples/tiny_transformer_classifier.py
    python examples/tiny_transformer_classifier.py --device cuda
    python examples/tiny_transformer_classifier.py --epochs 20 --lr 0.001
"""

import argparse
import time

import numpy as np

import orthon as ortho
import orthon.nn as nn
import orthon.optim as optim


# Dataset.

def make_dataset(n, vocab_size, seq_len, special_token, rng):
    """Return balanced (tokens [n,T], labels [n]) int64 arrays."""
    half = n // 2
    # Positive half: guarantee special_token appears at a random position.
    pos_tokens = rng.integers(0, vocab_size, size=(half, seq_len), dtype=np.int64)
    insert_col = rng.integers(0, seq_len, size=half)
    pos_tokens[np.arange(half), insert_col] = special_token
    # Accidental special tokens at other positions are harmless: the label is
    # 1 regardless, so no cleanup is needed on the positive half.
    pos_labels = np.ones(half, dtype=np.int64)

    # Negative half: no special_token.
    neg_tokens = rng.integers(0, vocab_size, size=(half, seq_len), dtype=np.int64)
    # Replace any accidental hits with token 0 (which is != special_token).
    neg_tokens[neg_tokens == special_token] = 0
    neg_labels = np.zeros(half, dtype=np.int64)

    tokens = np.concatenate([pos_tokens, neg_tokens], axis=0)
    labels = np.concatenate([pos_labels, neg_labels], axis=0)
    perm = rng.permutation(n)
    return tokens[perm], labels[perm]


# Model.

class TinyTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, num_heads,
                 num_classes, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(seq_len, embed_dim)
        self.block = nn.TransformerEncoderBlock(
            embed_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout
        )
        self.ln = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def _get_positions(self, B, T, device):
        pos_np = np.tile(np.arange(T, dtype=np.int64), (B, 1))
        pos = ortho.from_numpy(pos_np)
        on_cuda = device == "cuda" if isinstance(device, str) else device.is_cuda()
        if on_cuda:
            pos = pos.to("cuda")
        return pos

    def forward(self, tokens):
        B = int(tokens.shape[0])
        T = int(tokens.shape[1])
        pos = self._get_positions(B, T, tokens.device)
        x = (ortho.embedding(tokens, self.token_embed.weight)
             + ortho.embedding(pos, self.pos_embed.weight))
        x = self.block(x)
        x = self.ln(x)
        pooled = x.mean(dim=1)
        logits = self.classifier(pooled)
        return ortho.log_softmax(logits, dim=-1)


# Data helpers.

def _to_tensor(arr, device):
    t = ortho.from_numpy(arr)
    return t.to("cuda") if device == "cuda" else t


def batch_iter(tokens, labels, batch_size, rng=None):
    n = len(tokens)
    idx = np.arange(n)
    if rng is not None:
        idx = rng.permutation(n)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield tokens[idx[start:end]], labels[idx[start:end]]


# Train and eval steps.

def accuracy(log_probs, targets):
    lp = log_probs.to("cpu") if log_probs.device.is_cuda() else log_probs
    pred = [int(np.argmax(row)) for row in np.array(lp.tolist()).reshape(-1, 2)]
    tgt = targets.to("cpu") if targets.device.is_cuda() else targets
    tgt_list = tgt.tolist()
    correct = sum(p == int(t) for p, t in zip(pred, tgt_list))
    return correct / len(pred)


def run_epoch(model, optimizer, tokens, labels, batch_size, device, rng, train=True):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    for tok_batch, lbl_batch in batch_iter(tokens, labels, batch_size, rng if train else None):
        x = _to_tensor(tok_batch, device)
        y = _to_tensor(lbl_batch, device)

        if train:
            optimizer.zero_grad()

        with (ortho.no_grad() if not train else _noop_ctx()):
            log_probs = model(x)
            loss = ortho.nll_loss(log_probs, y, reduction="mean")

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy(log_probs, y)
        n_batches += 1

    return total_loss / n_batches, total_acc / n_batches


class _noop_ctx:
    def __enter__(self): return self
    def __exit__(self, *a): pass


# Main.

def main():
    parser = argparse.ArgumentParser(description="Tiny Transformer classifier (Orthon)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-size", type=int, default=2048)
    parser.add_argument("--test-size", type=int, default=512)
    args = parser.parse_args()

    vocab_size = 32
    seq_len = 8
    num_classes = 2
    special_token = 7

    rng = np.random.default_rng(args.seed)
    train_tokens, train_labels = make_dataset(
        args.train_size, vocab_size, seq_len, special_token, rng
    )
    test_tokens, test_labels = make_dataset(
        args.test_size, vocab_size, seq_len, special_token, rng
    )

    device = args.device
    if device == "cuda" and not ortho.cuda_enabled():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    model = TinyTransformerClassifier(
        vocab_size=vocab_size,
        seq_len=seq_len,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_classes=num_classes,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    )
    if device == "cuda":
        model.to("cuda")

    optimizer = optim.Adam(list(model.parameters()), lr=args.lr)
    train_rng = np.random.default_rng(args.seed + 1)

    n_params = sum(int(np.prod(list(p.shape))) for p in model.parameters())
    print(f"Model: {n_params} parameters | device={device} | "
          f"embed_dim={args.embed_dim} | num_heads={args.num_heads}")
    print(f"Task: detect token {special_token} in seq_len={seq_len} sequences "
          f"(vocab_size={vocab_size}, balanced binary)")
    print(f"{'epoch':>5} {'tr_loss':>8} {'tr_acc':>7} {'te_loss':>8} {'te_acc':>7} {'secs':>6}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(
            model, optimizer, train_tokens, train_labels,
            args.batch_size, device, train_rng, train=True
        )
        te_loss, te_acc = run_epoch(
            model, optimizer, test_tokens, test_labels,
            args.batch_size, device, None, train=False
        )
        elapsed = time.time() - t0
        print(f"{epoch:>5} {tr_loss:>8.4f} {tr_acc:>7.3f} {te_loss:>8.4f} {te_acc:>7.3f} {elapsed:>6.2f}s")


if __name__ == "__main__":
    main()
