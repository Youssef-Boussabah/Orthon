"""Small supervised-training helpers for Orthon examples."""


def _targets_to_list(targets):
    if hasattr(targets, "tolist"):
        return [int(v) for v in targets.tolist()]
    return [int(v) for v in targets]


def classification_accuracy(logits, targets):
    """Return classification accuracy for [N, C] logits and integer targets."""
    target_list = _targets_to_list(targets)
    values = logits.tolist()
    num_classes = int(logits.shape[1])
    correct = 0

    for i, target in enumerate(target_list):
        row = values[i * num_classes:(i + 1) * num_classes]
        pred = max(range(num_classes), key=lambda j: row[j])
        correct += int(pred == target)

    return correct / len(target_list)


def train_step(model, batch_x, batch_y, loss_fn, optimizer):
    """Run one supervised classification training step."""
    targets = _targets_to_list(batch_y)
    model.train()
    optimizer.zero_grad()

    logits = model(batch_x)
    loss = loss_fn(logits, targets)
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "accuracy": classification_accuracy(logits, targets),
        "num_examples": len(targets),
    }


def eval_step(model, batch_x, batch_y, loss_fn):
    """Run one supervised classification evaluation step."""
    targets = _targets_to_list(batch_y)
    model.eval()

    logits = model(batch_x)
    loss = loss_fn(logits, targets)

    return {
        "loss": float(loss.item()),
        "accuracy": classification_accuracy(logits, targets),
        "num_examples": len(targets),
    }


def weighted_metric_average(step_outputs):
    """Average step metrics weighted by each step's example count."""
    total = sum(row["num_examples"] for row in step_outputs)
    if total == 0:
        raise ValueError("weighted_metric_average() requires at least one example")

    return {
        "loss": sum(row["loss"] * row["num_examples"] for row in step_outputs) / total,
        "accuracy": sum(row["accuracy"] * row["num_examples"] for row in step_outputs) / total,
        "num_examples": total,
    }
