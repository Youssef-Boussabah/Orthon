"""Neural-network API demo.

Demonstrates: Sequential, Linear, ReLU, Conv2d, Module.to(),
              cross_entropy, Adam, state_dict save/load.

Everything runs on one device, selected once below. Mixed-device operands are
rejected by the runtime, so every tensor built here is placed on `device`
explicitly rather than relying on a default.

Run from repo root:
    python examples/nn_api_demo.py
"""
import tempfile
import os
import orthon
from orthon.nn import Conv2d, Linear, ReLU, Sequential
from orthon.optim import Adam

# Device selection.
device = "cuda" if orthon.cuda_enabled() else "cpu"
print(f"Device: {device}")

# Build a small model.
# Input: [N, 1, 8, 8] grayscale patches → 10 classes
model = Sequential(
    Conv2d(1, 4, kernel_size=3, padding=1),   # [N, 4, 8, 8]
    ReLU(),
    Conv2d(4, 8, kernel_size=3, padding=1),   # [N, 8, 8, 8]
    ReLU(),
)
# Flatten 8*8*8 = 512 and classify (via a separate Linear pass)
clf = Sequential(
    Linear(512, 64),
    ReLU(),
    Linear(64, 10),
)

print(f"\nConv backbone:\n{model}")
print(f"\nClassifier:\n{clf}")

# Move to device.
model.to(device)
clf.to(device)

# Synthetic batch, built on the same device as the model.
batch_size = 4
x = orthon.full([batch_size, 1, 8, 8], 0.1).to(device)
targets = list(range(batch_size))

# Forward pass.
feat = model(x)                          # [4, 8, 8, 8]
print(f"\nFeature map shape: {feat.shape}")

# Flatten spatial dims: [N, 8*8*8] = [N, 512]
feat_flat = feat.reshape([batch_size, 512])
logits = clf(feat_flat)
print(f"Logits shape:      {logits.shape}")

# Cross-entropy loss.
loss = orthon.cross_entropy(logits, targets)
print(f"Initial loss:      {loss.item():.4f}")

# Backward pass.
loss.backward()

# Adam step.
all_params = list(model.parameters()) + list(clf.parameters())
optimizer = Adam(all_params, lr=0.01)
optimizer.step()

# A few training steps.
print("\nTraining 20 steps with Adam:")
for step in range(20):
    optimizer.zero_grad()
    feat = model(x)
    feat_flat = feat.reshape([batch_size, 512])
    logits = clf(feat_flat)
    loss = orthon.cross_entropy(logits, targets)
    loss.backward()
    optimizer.step()
    if step % 5 == 4:
        print(f"  step {step+1:2d}: loss = {loss.item():.4f}")

# Save state dict.
with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as f:
    path = f.name

orthon.save_state_dict(clf, path)
print(f"\nSaved clf state_dict to {path}")

# Load into a fresh model on the same device. Loading is
# destination-authoritative, so the values land on whatever devices clf2's
# parameters already occupy.
clf2 = Sequential(Linear(512, 64), ReLU(), Linear(64, 10))
clf2.to(device)
orthon.load_state_dict(clf2, path)
os.unlink(path)

# Verify outputs match, feeding both models the same on-device probe.
clf.zero_grad()
clf2.zero_grad()
probe = orthon.full([1, 512], 0.5).to(device)
out1 = clf(probe).tolist()
out2 = clf2(probe).tolist()
match = all(abs(a - b) < 1e-5 for a, b in zip(out1, out2))
print(f"save/load roundtrip verified: {match}")

print("\nDone.")
