#include <orthon/autograd/engine.h>
#include <orthon/autograd/grad_mode.h>
#include <orthon/autograd/node.h>
#include <orthon/core/tensor_impl.h>
#include <orthon/dispatch/dispatcher.h>
#include <orthon/ops/binary_ops.h>
#include <orthon/ops/fill_ops.h>
#include <orthon/ops/shape_ops.h>
#include <orthon/runtime/runtime.h>
#include <deque>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace orthon {
namespace {
using Kind = GraphEdge::Kind;

[[noreturn]] void freed_history() {
    throw std::runtime_error(
        "Cannot call backward twice on a freed graph. Use retain_graph=True.");
}

Tensor accumulate_grad(const Tensor& existing, const Tensor& incoming) {
    return add(contiguous(existing), contiguous(incoming));
}

void preflight_leaf(const Tensor& leaf) {
    // Accumulation normalizes layouts before addition. Resolve that exact
    // execution profile without invoking the selected kernel or copying data.
    auto request = dispatch::unary_request(dispatch::OperatorId::Add, leaf);
    request.layout = leaf.numel() == 0 ? dispatch::LayoutClass::Empty
                                     : dispatch::LayoutClass::Contiguous;
    try {
        dispatch::default_dispatcher().resolve<dispatch::BinaryKernel>(request, default_runtime());
    } catch (const dispatch::UnsupportedOperation&) {
        throw std::runtime_error("backward(): autograd gradient accumulation is unsupported for this dtype on " +
                                 device_to_string(leaf.device()));
    }
}

// Gradients and scheduling state never escape this invocation. Tensor handles
// keep every identity key alive, including after direct history is released.
struct GraphTask {
    struct Value {
        Tensor tensor;
        GraphEdge edge;
        size_t dependencies = 0;
        unsigned visit = 0;  // unseen, on discovery stack, finished
        std::optional<Tensor> gradient;
        std::vector<size_t> inputs;
    };
    std::vector<Value> values;
    std::unordered_map<TensorImpl*, size_t> identities;
    std::vector<size_t> topology;  // input-first, also preserves leaf commit order
    std::deque<size_t> ready;
    std::unordered_set<uintptr_t> finalizer_keys;
    std::vector<std::function<void()>> finalizers;
    bool retain_graph;

    explicit GraphTask(bool retain) : retain_graph(retain) {}

    size_t add_value(const Tensor& tensor, const GraphEdge& edge) {
        auto [it, inserted] = identities.emplace(tensor.impl().get(), values.size());
        if (inserted) {
            values.push_back({tensor, edge, 0, 0, {}, {}});
            for (const auto& entry : tensor.impl()->autograd_meta().backward_finalizers())
                if (finalizer_keys.insert(entry.first).second) finalizers.push_back(entry.second);
        }
        else if (values[it->second].edge.kind != edge.kind ||
                 values[it->second].edge.producer != edge.producer)
            throw std::runtime_error("backward(): inconsistent captured graph history");
        return it->second;
    }

    void discover(const Tensor& root) {
        const size_t root_index = add_value(root, GraphEdge::capture(root));
        struct Frame { size_t index; bool finish; };
        std::vector<Frame> stack{{root_index, false}};
        while (!stack.empty()) {
            const auto frame = stack.back();
            stack.pop_back();
            if (frame.finish) {
                values[frame.index].visit = 2;
                topology.push_back(frame.index);
                continue;
            }
            if (values[frame.index].visit == 2) continue;
            if (values[frame.index].visit == 1)
                throw std::runtime_error("backward(): cycle in captured graph");
            values[frame.index].visit = 1;
            stack.push_back({frame.index, true});
            const auto edge = values[frame.index].edge;
            if (edge.kind == Kind::Released) freed_history();
            if (edge.kind != Kind::Producer) continue;
            const auto& node = edge.producer;
            if (!node || !node->edges_sealed() || node->input_edges().size() != node->inputs.size())
                throw std::runtime_error("backward(): invalid captured graph edges");
            std::vector<size_t> input_indices;
            input_indices.reserve(node->inputs.size());
            for (size_t i = 0; i < node->inputs.size(); ++i) {
                const auto& input_edge = node->input_edges()[i];
                const size_t input = add_value(node->inputs[i], input_edge);
                input_indices.push_back(input);
                if (input_edge.kind != Kind::None) ++values[input].dependencies;
            }
            values[frame.index].inputs = std::move(input_indices);
            // Reverse push yields the old input-first discovery/commit order.
            for (auto it = values[frame.index].inputs.rbegin();
                 it != values[frame.index].inputs.rend(); ++it) {
                if (values[*it].edge.kind != Kind::None) stack.push_back({*it, false});
            }
        }
    }

    void preflight() const {
        for (size_t index : topology) {
            const auto& value = values[index];
            if (value.edge.producer) value.edge.producer->validate_saved_versions();
        }
        for (size_t index : topology) {
            const auto& value = values[index];
            if (value.edge.kind == Kind::Leaf && value.tensor.requires_grad())
                preflight_leaf(value.tensor);
        }
    }

    void evaluate(const Tensor& root) {
        Tensor seed = ones(root.shape(), root.dtype());
        if (root.device().is_cuda()) seed = seed.to(root.device());
        values[0].gradient = std::move(seed);
        ready.push_back(0);
        size_t completed = 0;
        while (!ready.empty()) {
            const size_t index = ready.front();
            ready.pop_front();
            auto& value = values[index];
            ++completed;
            if (value.edge.kind != Kind::Producer) continue;
            const auto& node = value.edge.producer;
            std::vector<Tensor> gradients;
            if (value.gradient) {
                node->validate_saved_versions();
                gradients = node->backward(*value.gradient);
            }
            for (size_t i = 0; i < value.inputs.size(); ++i) {
                auto& input = values[value.inputs[i]];
                if (input.edge.kind == Kind::None) continue;
                // Preserve the permissive ordered-prefix convention: NLL has
                // two inputs and returns only its differentiable first slot.
                if (i < gradients.size() && gradients[i].impl()) {
                    const Tensor& gradient = gradients[i];
                    if (gradient.shape() != input.tensor.shape() ||
                        gradient.dtype() != input.tensor.dtype() ||
                        gradient.device() != input.tensor.device())
                        throw std::runtime_error("backward(): invalid input gradient shape, dtype or device");
                    if (input.gradient)
                        input.gradient = accumulate_grad(*input.gradient, gradient);
                    else input.gradient = gradient;
                }
                if (--input.dependencies == 0) ready.push_back(value.inputs[i]);
            }
        }
        if (completed != topology.size())
            throw std::runtime_error("backward(): incomplete graph dependencies");
    }

    void commit_leaves() {
        std::unordered_set<const Buffer*> claimed;
        // A formula may return a view. Stored gradients must also be independent
        // of forward bytes, even when a diagnostic formula returns an input.
        for (const auto& value : values) claimed.insert(value.tensor.impl()->buffer().get());
        for (size_t index : topology) {
            auto& value = values[index];
            auto& tensor = value.tensor;
            if (value.edge.kind != Kind::Leaf || !tensor.is_leaf() ||
                !tensor.requires_grad() || !value.gradient) continue;
            Tensor candidate = tensor.has_grad() ? accumulate_grad(tensor.grad(), *value.gradient)
                                                : contiguous(*value.gradient);
            if (!claimed.insert(candidate.impl()->buffer().get()).second) {
                candidate = candidate.to(candidate.device());
                claimed.insert(candidate.impl()->buffer().get());
            }
            tensor.set_grad(candidate);
            tensor.impl()->run_post_accumulate_grad_hooks();
        }
    }

    void release_history() {
        if (retain_graph) return;
        for (size_t index : topology) {
            auto& value = values[index];
            if (value.edge.kind == Kind::Producer &&
                value.tensor.grad_fn() == value.edge.producer)
                value.tensor.clear_grad_fn();
        }
        // All producers stay owned until every reached direct history is clear.
        // Drop task ownership output-first to avoid recursive chain destruction.
        for (auto it = topology.rbegin(); it != topology.rend(); ++it)
            values[*it].edge.producer.reset();
    }
    void finalize_backward() {
        for (auto& callback : finalizers) callback();
    }
};
}  // namespace

void AutogradEngine::backward(const Tensor& root, bool retain_graph) const {
    if (!root.requires_grad())
        throw std::runtime_error("backward(): tensor does not require grad");
    if (!root.is_leaf() && !root.has_grad_fn()) freed_history();
    NoGradGuard backward_guard;
    GraphTask task(retain_graph);
    task.discover(root);
    task.preflight();
    task.evaluate(root);
    task.commit_leaves();
    task.finalize_backward();
    // Exceptions before here preserve direct histories; hook exceptions may
    // leave earlier commits changed, matching the established boundary.
    task.release_history();
}

const AutogradEngine& default_autograd_engine() {
    static const AutogradEngine engine;
    return engine;
}
}  // namespace orthon
