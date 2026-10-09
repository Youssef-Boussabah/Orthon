"""Current documentation scope and small parsers for repository references.

Inline links and reference definitions are checked outside fenced examples.
Anchors are excluded: file existence is the contract, not renderer slug rules.
Explicit inline-code paths exclude wildcards and conceptual placeholders.
"""
import json
import re
from urllib.parse import unquote, urlsplit

CURRENT_DOCUMENTS = (
    "README.md", "docs/README.md", "docs/api.md", "docs/architecture.md",
    "docs/autograd.md", "docs/cuda_runtime.md", "docs/cuda_coverage.md",
    "docs/training_stack.md", "docs/distributed.md", "docs/checkpoint_format.md",
    "docs/testing.md", "docs/training_workloads.md", "docs/benchmarking.md",
    "docs/nsight_profiling.md",
)
PUBLIC_NAMES = {
    "orthon": frozenset("""
        DType DeviceType Device Tensor empty zeros ones full from_numpy fill_
        add mul sub div reshape flatten transpose contiguous sum mean max argmax
        neg relu sigmoid tanh leaky_relu gelu exp dropout embedding sqrt reciprocal
        to cast matmul fused_linear_relu softmax log_softmax cross_entropy nll_loss
        conv2d maxpool2d layer_norm scaled_dot_product_attention isfinite_all
        cuda_enabled nccl_enabled cuda_synchronize is_grad_enabled set_grad_enabled
        no_grad manual_seed save_state_dict load_state_dict save_checkpoint
        load_checkpoint nn optim amp training
    """.split()),
    "orthon.nn": frozenset("""
        Module Parameter Linear FusedLinearReLU ReLU Sigmoid Tanh LeakyReLU GELU
        Flatten LayerNorm Embedding Sequential Conv2d MaxPool2d Dropout BatchNorm1d
        BatchNorm2d NLLLoss SelfAttention MultiHeadSelfAttention TransformerEncoderBlock
    """.split()),
    "orthon.optim": frozenset({"SGD", "Adam"}),
    "orthon.amp": frozenset({"MixedPrecisionOptimizer", "GradScaler"}),
    "orthon.distributed": frozenset("""
        Backend register_backend rank_config_from_env DistributedDataParallel DDP
        init_process_group destroy_process_group is_initialized get_rank
        get_world_size barrier all_reduce broadcast spawn Work
    """.split()),
    "orthon.training": frozenset({
        "classification_accuracy", "train_step", "eval_step", "weighted_metric_average",
    }),
}
RETIRED_TOKENS = (
    "src/ops/cpu/", "src/ops/cuda/", "orthon._memory_pool_", "orthon._matmul_",
    "cuda_conv2d_profile_reset", "cuda_conv2d_profile_summary", "ORTHON_PROFILE_CONV2D",
    "Module._replace_parameter", "orthon._scalar_like",
)
STALE_STREAM_PHRASES = (
    "all work runs on the default stream", "everything runs on the default stream",
    "no stream-aware allocation", "one private nccl stream per work",
    "async nccl uses a private work stream",
)


def fences(text, language):
    return re.findall(r"^```" + re.escape(language) + r"\s*\n(.*?)^```\s*$",
                      text, re.M | re.S)


def outside_fences(text):
    return re.sub(r"^```[^\n]*\n.*?^```\s*$", "", text, flags=re.M | re.S)


def internal_links(text):
    text = outside_fences(text)
    targets = re.findall(r'\[[^\]\n]+\]\((<[^>]+>|[^\s)]+)(?:\s+"[^"\n]*")?\)', text)
    targets += re.findall(r"^\s*\[[^\]]+\]:\s*(<[^>]+>|\S+)", text, re.M)
    for target in targets:
        parsed = urlsplit(target.strip("<>"))
        if not parsed.scheme and not parsed.netloc and parsed.path:
            yield unquote(parsed.path)


def source_paths(text):
    pattern = r"(?<!`)`((?:include|src|bindings|python|tests|examples|benchmarks|docs)/[^`\s]+)`(?!`)"
    for path in re.findall(pattern, text):
        if not any(c in path for c in "*<>|{}"):
            yield path


def command_paths(text):
    for module in re.findall(r"python\s+-m\s+((?:benchmarks|examples|orthon\.distributed)\.[\w.]+)", text):
        prefix = "python/" if module.startswith("orthon.") else ""
        yield prefix + module.replace(".", "/") + ".py"
    yield from re.findall(r"python\s+((?:examples|benchmarks|tests)/[\w/.-]+\.py)", text)


def api_manifest(text):
    blocks = fences(text, "json")
    if len(blocks) != 1:
        raise ValueError("API index must contain one JSON public-name manifest")
    return {module: frozenset(names) for module, names in json.loads(blocks[0]).items()}
