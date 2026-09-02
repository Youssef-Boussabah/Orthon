"""Tiny causal language model demo using Orthon.

Synthetic task: next-token prediction on arithmetic progressions mod vocab_size.
Each training sequence is s, s+d, s+2d, ... mod vocab_size with d in {1, 2, 3}.
The model learns to predict the next token given the causal context.

Usage:
    python examples/tiny_causal_lm.py
    python examples/tiny_causal_lm.py --device cuda --epochs 30
    python examples/tiny_causal_lm.py --device cuda --epochs 30 \\
        --embed-dim 32 --num-heads 4 --num-layers 1 --lr 0.001
"""

import argparse
import time

import numpy as np

import orthon as ortho
import orthon.nn as nn
import orthon.optim as optim


# Causal mask.

def make_causal_mask(seq_len, device="cpu"):
    """Return an additive causal mask of shape [1, seq_len, seq_len].

    Allowed positions (j <= i) get 0.0; future positions (j > i) get -1e9.
    Broadcasting over the batch*heads dimension of attention scores works
    because attention scores have shape [B*H, T, T] and this mask is [1, T, T].
    """
    # np.triu(X, k=1): returns X with elements on or below the main diagonal
    # set to zero, keeping only j > i, exactly the positions to mask.
    mask_np = np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)
    mask = ortho.from_numpy(mask_np.reshape(1, seq_len, seq_len))
    if device == "cuda":
        mask = mask.to("cuda")
    return mask


# Dataset: arithmetic progressions mod vocab_size.

def make_dataset(n, vocab_size, block_size, rng):
    """Return (inputs [n, block_size], targets [n, block_size]) int64 arrays.

    Each row is a sliding window over an arithmetic progression:
        seq = s, s+d, s+2d, ..., s+block_size*d  (mod vocab_size)
    input  = seq[:block_size]   (context tokens)
    target = seq[1:]            (next tokens for each position)
    """
    steps = [1, 2, 3]
    inputs  = np.empty((n, block_size), dtype=np.int64)
    targets = np.empty((n, block_size), dtype=np.int64)
    for i in range(n):
        s = int(rng.integers(0, vocab_size))
        d = steps[int(rng.integers(0, len(steps)))]
        seq = [(s + d * j) % vocab_size for j in range(block_size + 1)]
        inputs[i]  = seq[:block_size]
        targets[i] = seq[1:]
    return inputs, targets


# Model.

class TinyCausalLM(nn.Module):
    """Tiny causal language model: embeddings + TransformerEncoderBlock(s) + LM head.

    Each TransformerEncoderBlock receives the causal additive mask so attention
    never looks at future positions. The lm_head projects each token position
    to logits over the vocabulary.
    """

    def __init__(self, vocab_size, block_size, embed_dim, num_heads,
                 num_layers=1, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.num_layers = num_layers

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed   = nn.Embedding(block_size, embed_dim)

        for i in range(num_layers):
            setattr(self, f"block_{i}",
                    nn.TransformerEncoderBlock(embed_dim, num_heads,
                                              mlp_ratio=mlp_ratio,
                                              dropout=dropout))

        self.ln_f    = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx, mask=None):
        B = int(idx.shape[0])
        T = int(idx.shape[1])

        pos_np = np.tile(np.arange(T, dtype=np.int64), (B, 1))
        pos = ortho.from_numpy(pos_np)
        if idx.device.is_cuda():
            pos = pos.to("cuda")

        x = (ortho.embedding(idx, self.token_embed.weight)
             + ortho.embedding(pos, self.pos_embed.weight))

        for i in range(self.num_layers):
            x = getattr(self, f"block_{i}")(x, mask=mask)

        x = self.ln_f(x)

        # lm_head is 2D-only: flatten [B, T, E] → [B*T, E], project, reshape back
        E = int(x.shape[2])
        logits_flat    = self.lm_head(x.reshape([B * T, E]))   # [B*T, V]
        V              = int(logits_flat.shape[1])
        log_probs_flat = ortho.log_softmax(logits_flat, dim=-1) # [B*T, V]
        return log_probs_flat.reshape([B, T, V])                # [B, T, V]


# Data helpers.

def _to_tensor(arr, device):
    t = ortho.from_numpy(arr)
    return t.to("cuda") if device == "cuda" else t


def batch_iter(inputs, targets, batch_size, rng=None):
    n   = len(inputs)
    idx = rng.permutation(n) if rng is not None else np.arange(n)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sl  = idx[start:end]
        yield inputs[sl], targets[sl]


# Train and eval.

def run_epoch(model, optimizer, mask, inputs, targets, batch_size, device,
              rng=None, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = 0

    for inp_batch, tgt_batch in batch_iter(inputs, targets, batch_size,
                                           rng if train else None):
        B = len(inp_batch)
        T = inp_batch.shape[1]
        V = model.vocab_size

        x = _to_tensor(inp_batch, device)
        y = _to_tensor(tgt_batch, device)

        if train:
            optimizer.zero_grad()
            log_probs = model(x, mask=mask)                    # [B, T, V]
            loss = ortho.nll_loss(
                log_probs.reshape([B * T, V]),
                y.reshape([B * T]),
                reduction="mean",
            )
            loss.backward()
            optimizer.step()
        else:
            with ortho.no_grad():
                log_probs = model(x, mask=mask)                # [B, T, V]
                loss = ortho.nll_loss(
                    log_probs.reshape([B * T, V]),
                    y.reshape([B * T]),
                    reduction="mean",
                )

        total_loss += loss.item()

        # Token accuracy: native argmax on device; only bring [B*T] int64 to CPU.
        pred_flat = ortho.argmax(log_probs.reshape([B * T, V]), dim=-1)
        if pred_flat.device.is_cuda():
            pred_flat = pred_flat.to("cpu")
        pred_np = np.array(pred_flat.tolist(), dtype=np.int64)
        total_acc += float((pred_np == tgt_batch.reshape(-1)).mean())
        n_batches += 1

    return total_loss / n_batches, total_acc / n_batches


# Autoregressive generation.

def generate(model, start_tokens, max_new_tokens, device):
    """Greedy autoregressive generation.

    Repeatedly feeds the last block_size tokens through the model with a
    fresh causal mask, samples the argmax at the final position, and appends
    the predicted token to the running context.
    """
    model.eval()
    context    = list(start_tokens)
    block_size = model.block_size

    for _ in range(max_new_tokens):
        crop = context[-block_size:]
        T    = len(crop)

        inp  = _to_tensor(np.array([crop], dtype=np.int64), device)
        mask = make_causal_mask(T, device)

        with ortho.no_grad():
            log_probs = model(inp, mask=mask)   # [1, T, V]

        next_token = ortho.argmax(log_probs[0][T - 1]).item()
        context.append(next_token)

    return context


# Main.

def main():
    parser = argparse.ArgumentParser(description="Tiny causal LM demo (Orthon)")
    parser.add_argument("--device",          default="cuda")
    parser.add_argument("--epochs",          type=int,   default=30)
    parser.add_argument("--batch-size",      type=int,   default=64)
    parser.add_argument("--vocab-size",      type=int,   default=32)
    parser.add_argument("--block-size",      type=int,   default=16)
    parser.add_argument("--embed-dim",       type=int,   default=32)
    parser.add_argument("--num-heads",       type=int,   default=4)
    parser.add_argument("--num-layers",      type=int,   default=1)
    parser.add_argument("--mlp-ratio",       type=float, default=4.0)
    parser.add_argument("--dropout",         type=float, default=0.0)
    parser.add_argument("--lr",              type=float, default=0.001)
    parser.add_argument("--seed",            type=int,   default=0)
    parser.add_argument("--train-size",      type=int,   default=2048)
    parser.add_argument("--test-size",       type=int,   default=512)
    parser.add_argument("--generate-tokens", type=int,   default=16)
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not ortho.cuda_enabled():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    rng = np.random.default_rng(args.seed)
    train_inp, train_tgt = make_dataset(
        args.train_size, args.vocab_size, args.block_size, rng)
    test_inp,  test_tgt  = make_dataset(
        args.test_size,  args.vocab_size, args.block_size, rng)

    model = TinyCausalLM(
        vocab_size  = args.vocab_size,
        block_size  = args.block_size,
        embed_dim   = args.embed_dim,
        num_heads   = args.num_heads,
        num_layers  = args.num_layers,
        mlp_ratio   = args.mlp_ratio,
        dropout     = args.dropout,
    )
    if device == "cuda":
        model.to("cuda")

    causal_mask = make_causal_mask(args.block_size, device)
    optimizer   = optim.Adam(list(model.parameters()), lr=args.lr)
    train_rng   = np.random.default_rng(args.seed + 1)

    n_params = sum(int(np.prod(list(p.shape))) for p in model.parameters())
    print(f"Model: {n_params} parameters | device={device} | "
          f"embed_dim={args.embed_dim} | num_heads={args.num_heads} | "
          f"num_layers={args.num_layers}")
    print(f"Task : next-token prediction on arithmetic progressions "
          f"(vocab={args.vocab_size}, block_size={args.block_size}, steps={{1,2,3}})")
    print(f"{'epoch':>5} {'tr_loss':>8} {'tr_acc':>7} "
          f"{'te_loss':>8} {'te_acc':>7} {'secs':>6}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(
            model, optimizer, causal_mask,
            train_inp, train_tgt, args.batch_size, device, train_rng, train=True)
        te_loss, te_acc = run_epoch(
            model, optimizer, causal_mask,
            test_inp,  test_tgt,  args.batch_size, device, None,      train=False)
        print(f"{epoch:>5} {tr_loss:>8.4f} {tr_acc:>7.3f} "
              f"{te_loss:>8.4f} {te_acc:>7.3f} {time.time()-t0:>6.2f}s")

    # --- Autoregressive generation demo ---
    # Prompt is the first 4 tokens of a step-2 progression starting at 3.
    prompt    = [3, 5, 7, 9]
    generated = generate(model, prompt, args.generate_tokens, device)
    expected  = [(3 + 2 * k) % args.vocab_size for k in range(len(generated))]
    print(f"\n--- Autoregressive generation (step-2 progression from {prompt[0]}) ---")
    print(f"Prompt   : {prompt}")
    print(f"Generated: {generated[len(prompt):]}")
    print(f"Expected : {expected[len(prompt):]}")
    print(f"Full seq : {generated}")


if __name__ == "__main__":
    main()
