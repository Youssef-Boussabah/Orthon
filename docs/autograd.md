# Autograd

Orthon implements first-order reverse-mode differentiation. Native graph contracts
live in `include/orthon/autograd/node.h`, formulas in the autograd node headers,
and execution in `src/autograd/engine.cpp`. `Tensor::backward` is a thin delegation
to the default AutogradEngine, not a second traversal implementation.

## Graph identity and captured history

Every TensorImpl has its own AutogradMeta. Requires-grad, leaf identity, stored
gradient and direct producer history belong to that identity even when views
share a Buffer. A Node retains input Tensor handles and sealed GraphEdges.
An edge captures one of `None`, `Leaf`, `Producer` or `Released`; direct history
is followed from the root at invocation start. There is no global graph registry.

A GraphTask retains Tensor handles to keep its TensorImpl identity keys alive.
Iterative discovery builds an identity map, ordered input indices, dependency
counts, an input-first topology and a ready queue. It rejects cycles, inconsistent
captured histories and invalid sealed edges. Task state does not escape backward.

## Execution and failure boundaries

1. Validate the root requires grad and has usable history. Calling backward on a
   released nonleaf raises the freed-graph error.
2. Enter one NoGradGuard covering the whole invocation. Backward formulas and
   accumulation do not build a higher-order graph.
3. Discover the captured graph and preflight every reached producer's saved
   versions, then the declared dispatch profile for each differentiable leaf's
   gradient accumulation. These preflight failures preserve existing leaf
   gradients and direct history.
4. Seed the root with ones of its shape/dtype on its exact device. Backward of a
   nonscalar therefore uses an all-ones seed; there is no public gradient argument.
5. Evaluate dependency-ready producers, rechecking saved versions before each
   formula. Validate returned input gradients against input shape, dtype and exact
   device. Contributions accumulate per tensor identity. An ordered prefix of
   differentiable input slots is permitted, as used by NLL's target input.
6. After evaluation, commit leaf gradients in input-first order. Existing gradients
   accumulate across backward calls. Stored gradients use independent Buffers,
   including when a formula returns a view or two leaves receive the same value.
   Run each leaf's post-accumulate hooks after its commit.
7. Run unique private backward finalizers collected from reached AutogradMeta,
   once per key in the task, after leaf commits/hooks and before graph release.
8. Unless `retain_graph=True`, clear reached direct producer histories, then drop
   task producer ownership output-first to avoid recursive chain destruction.

This boundary does not promise a fully transactional backward for arbitrary
exceptions: leaf hooks/finalizers can fail after earlier leaf commits. Exceptions
before final history release preserve direct history. Retaining a graph permits
another backward but still checks saved mutation generations each time.

## Buffer mutation watches

Saved-version watches retain a Buffer and its generation, plus a diagnostic role.
The Buffer is the shared physical alias identity; a write through any alias is
visible to every watch. Watches are selective: formulas watch the values they
read, rather than every input indiscriminately. Value-insensitive transfer/cast
backward does not need to watch source values.

A successful same-value write still counts. Transpose, reshape and other metadata
operations do not count as writes. Version checks protect backward from stale
forward values; they are not a general concurrent-mutation protocol.

## Views, copies and accumulation

View nodes route gradients to the parent's logical coordinates. Slice/index
backward scatters into the parent shape; transpose reverses its dimensions;
reshape restores input geometry. Contiguous materialization passes gradients
through to the original identity. Floating-to-floating cast sends gradients back
in the source dtype; integer/bool conversions do not join a graph. Transfer sends
gradients to the source device. Same-dtype cast is contiguity normalization.

A leaf's `.grad` is initially None. `zero_grad()` clears it; backward commits a
new independent accumulator when a contribution exists. Use explicit grad clearing
for training steps. `set_grad()` is a low-level public assignment operation whose
shape/dtype/device profile must remain suitable for consumers.

```python
import orthon as o

x = o.full([2], 3.0).requires_grad_(True)
loss = o.sum(x * x)
loss.backward(retain_graph=True)
print(x.grad.tolist())
loss.backward()
print(x.grad.tolist())
x.zero_grad()
```

Private finalizers serve DDP's Reducer and are not a public Python hook API.
They move deterministic distributed scheduling to backward completion, after
participation has been observed. See [distributed](distributed.md) and
[training](training_stack.md). Autograd support also depends on the
[operator/dtype profile](cuda_coverage.md); allocation support alone does not
imply supported gradient accumulation.
