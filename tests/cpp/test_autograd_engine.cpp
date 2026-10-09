#include <gtest/gtest.h>
#include <orthon/autograd/engine.h>
#include <orthon/autograd/grad_mode.h>
#include <orthon/autograd/node.h>
#include <orthon/core/tensor_impl.h>
#include <orthon/ops/binary_ops.h>
#include <orthon/ops/cast_ops.h>
#include <orthon/ops/fill_ops.h>
#include <orthon/ops/reduce_ops.h>
#include <orthon/ops/scalar_ops.h>
#include <orthon/ops/shape_ops.h>
#include <future>
#include <stdexcept>

using namespace orthon;
namespace {
float first(const Tensor& t) { return static_cast<const float*>(t.data())[0]; }
Tensor leaf(float value = 2) {
    auto x = full({2}, value);
    x.requires_grad_(true);
    return x;
}

class DiagnosticNode : public Node {
public:
    explicit DiagnosticNode(std::vector<Tensor> in) { inputs = std::move(in); }
    int calls = 0;
    bool fail = false;
    bool malformed = false;
    bool prefix = false;
    bool null_slot = false;
    bool forward_alias = false;
    std::function<void()> callback;
    std::vector<Tensor> backward(const Tensor& gradient) override {
        ++calls;
        EXPECT_FALSE(is_grad_enabled());
        if (callback) callback();
        if (fail) throw std::runtime_error("injected formula failure");
        if (malformed) return {ones({3})};
        if (null_slot) return {Tensor(nullptr)};
        if (forward_alias) return {inputs[0]};
        if (prefix) return {gradient};
        return std::vector<Tensor>(inputs.size(), gradient);
    }
};

Tensor attach(const std::shared_ptr<DiagnosticNode>& node) {
    auto out = ones({2});
    out.requires_grad_(true);
    out.set_is_leaf(false);
    out.set_grad_fn(node);
    return out;
}
}  // namespace

TEST(AutogradEngineFinalizer, DeduplicatesKeysAfterEveryLeafHookAndReplacesRegistration) {
    auto a = leaf(), b = leaf();
    int hooks = 0, calls = 0;
    a.add_post_accumulate_grad_hook([&] { ++hooks; });
    b.add_post_accumulate_grad_hook([&] { ++hooks; });
    a.register_backward_finalizer(7, [&] { FAIL() << "replaced finalizer"; });
    auto finalizer = [&] {
        ++calls;
        EXPECT_EQ(hooks, 2 * calls);
        EXPECT_TRUE(a.has_grad()); EXPECT_TRUE(b.has_grad());
    };
    a.register_backward_finalizer(7, finalizer);
    b.register_backward_finalizer(7, finalizer);
    auto root = add(a, b);
    root.register_backward_finalizer(7, finalizer);
    root.backward(true); root.backward();
    EXPECT_EQ(calls, 2);
    EXPECT_FLOAT_EQ(first(a.grad()), 2);
}
TEST(AutogradEngineFinalizer, EvaluationAndSavedVersionFailuresNeverFinalize) {
    auto a = leaf();
    int calls = 0;
    a.register_backward_finalizer(8, [&] { ++calls; });
    auto node = std::make_shared<DiagnosticNode>(std::vector<Tensor>{a});
    node->fail = true;
    auto root = attach(node);
    EXPECT_THROW(root.backward(), std::runtime_error);
    EXPECT_EQ(calls, 0); EXPECT_FALSE(a.has_grad());
    node->fail = false;
    root.backward(); EXPECT_EQ(calls, 1);
    auto square = mul(a, a);
    a.impl()->buffer()->bump_version();
    EXPECT_THROW(square.backward(), std::runtime_error);
    EXPECT_EQ(calls, 1);
}

TEST(AutogradEngine, DiamondFiveConsumersAndDuplicateSlotsRunProducerOnce) {
    auto x = leaf();
    auto node = std::make_shared<DiagnosticNode>(std::vector<Tensor>{x});
    auto shared = attach(node);
    auto root = add(shared, shared);
    for (int i = 1; i <= 5; ++i) root = add(root, scalar_mul(shared, i));
    root.backward(true);
    EXPECT_EQ(node->calls, 1);
    EXPECT_FLOAT_EQ(first(x.grad()), 17);
    root.backward();
    EXPECT_EQ(node->calls, 2);
    EXPECT_FLOAT_EQ(first(x.grad()), 34);
    EXPECT_FALSE(shared.has_grad());
}

TEST(AutogradEngine, RecurrentDagRunsEachProducerOnce) {
    auto x = leaf();
    auto root = x;
    std::vector<std::shared_ptr<DiagnosticNode>> nodes;
    for (int i = 0; i < 6; ++i) {
        auto node = std::make_shared<DiagnosticNode>(std::vector<Tensor>{root, root, x});
        root = attach(node);
        nodes.push_back(node);
    }
    root.backward();
    EXPECT_FLOAT_EQ(first(x.grad()), 127);
    for (const auto& node : nodes) EXPECT_EQ(node->calls, 1);
}

TEST(AutogradEngineLifetime, LargeFanoutDagCombinesBeforeOneEvaluationAndDestroys) {
    auto x = leaf();
    auto node = std::make_shared<DiagnosticNode>(std::vector<Tensor>{x});
    std::weak_ptr<Node> predecessor = node;
    {
        auto shared = attach(node);
        auto root = scalar_mul(shared, 1);
        int coefficient = 1;
        for (int i = 1; i <= 512; ++i) {
            const int scale = 1 + i % 7;
            coefficient += scale;
            root = add(root, scalar_mul(shared, scale));
        }
        root.backward();
        EXPECT_EQ(node->calls, 1);
        EXPECT_FLOAT_EQ(first(x.grad()), static_cast<float>(coefficient));
        EXPECT_FALSE(root.has_grad_fn());
    }
    node.reset();
    EXPECT_TRUE(predecessor.expired());
}

TEST(AutogradEngineLifetime, SiblingKeepsSharedProducerAfterDirectRelease) {
    auto x = leaf();
    auto shared = mul(x, x);
    std::weak_ptr<Node> predecessor = shared.grad_fn();
    auto a = scalar_mul(shared, 2);
    auto b = scalar_mul(shared, 3);
    std::weak_ptr<Node> first_root = a.grad_fn();
    a.backward();
    EXPECT_FALSE(a.has_grad_fn());
    EXPECT_EQ(a.impl()->autograd_meta().history(), AutogradMeta::History::Released);
    EXPECT_FALSE(shared.has_grad_fn());
    EXPECT_TRUE(first_root.expired());
    EXPECT_FALSE(predecessor.expired());
    b.backward();
    EXPECT_FLOAT_EQ(first(x.grad()), 20);
    EXPECT_TRUE(predecessor.expired());
}

TEST(AutogradEngineLifetime, RetainedRootSurvivesSiblingAndFinalReleaseDestroysNodes) {
    auto x = leaf();
    auto shared = mul(x, x);
    std::weak_ptr<Node> predecessor = shared.grad_fn();
    auto a = scalar_mul(shared, 2);
    auto b = scalar_mul(shared, 3);
    a.backward(true);
    b.backward();
    EXPECT_TRUE(a.has_grad_fn());
    EXPECT_FALSE(shared.has_grad_fn());
    EXPECT_FALSE(predecessor.expired());
    a.backward();
    EXPECT_FLOAT_EQ(first(x.grad()), 28);
    EXPECT_TRUE(predecessor.expired());
}

TEST(AutogradEngineLifetime, DroppingFinalRootDestroysUnreleasedGraphWithoutCycle) {
    auto x = leaf();
    std::weak_ptr<Node> predecessor;
    std::weak_ptr<Node> root_node;
    {
        auto shared = mul(x, x);
        predecessor = shared.grad_fn();
        auto root = scalar_mul(shared, 2);
        root_node = root.grad_fn();
    }
    EXPECT_TRUE(predecessor.expired());
    EXPECT_TRUE(root_node.expired());
    EXPECT_FALSE(x.has_grad());
}

TEST(AutogradEngineLifetime, RetentionKeepsTopologyUntilRootDropped) {
    auto x = leaf();
    std::weak_ptr<Node> predecessor;
    {
        auto shared = mul(x, x);
        predecessor = shared.grad_fn();
        auto root = scalar_mul(shared, 2);
        root.backward(true);
        EXPECT_FALSE(predecessor.expired());
        EXPECT_TRUE(shared.has_grad_fn());
    }
    EXPECT_TRUE(predecessor.expired());
}

TEST(AutogradEngine, HooksOncePerContributingLeafAfterCombinedCommitAndAgainOnRetention) {
    auto x = leaf(), y = leaf(), unused = leaf();
    int x_hooks = 0, y_hooks = 0, unused_hooks = 0;
    x.add_post_accumulate_grad_hook([&] {
        ++x_hooks;
        EXPECT_FLOAT_EQ(first(x.grad()), 3 * x_hooks);
        EXPECT_FALSE(is_grad_enabled());
        EXPECT_FALSE(x.grad().requires_grad());
    });
    y.add_post_accumulate_grad_hook([&] {
        ++y_hooks;
        EXPECT_FLOAT_EQ(first(y.grad()), static_cast<float>(y_hooks));
    });
    unused.add_post_accumulate_grad_hook([&] { ++unused_hooks; });
    auto root = add(add(x, x), add(x, y));
    root.backward(true);
    root.backward();
    EXPECT_EQ(x_hooks, 2);
    EXPECT_EQ(y_hooks, 2);
    EXPECT_EQ(unused_hooks, 0);
}

TEST(AutogradEngine, SavedVersionPreflightPreservesOldGradNoneAndNoHooks) {
    auto x = leaf(), y = leaf();
    x.backward();
    int hooks = 0;
    x.add_post_accumulate_grad_hook([&] { ++hooks; });
    y.add_post_accumulate_grad_hook([&] { ++hooks; });
    auto sensitive = mul(x, y);
    auto root = sensitive;
    for (int i = 0; i < 64; ++i) root = scalar_add(root, 1);
    mul_(x, 2);
    EXPECT_THROW(root.backward(), std::runtime_error);
    EXPECT_FLOAT_EQ(first(x.grad()), 1);
    EXPECT_FALSE(y.has_grad());
    EXPECT_EQ(hooks, 0);
    EXPECT_TRUE(root.has_grad_fn());
    EXPECT_TRUE(is_grad_enabled());
}

TEST(AutogradEngine, VersionsRecheckedImmediatelyBeforeFormula) {
    auto x = leaf();
    auto saved = std::make_shared<DiagnosticNode>(std::vector<Tensor>{x});
    saved->watch_version(x, "input");
    auto shared = attach(saved);
    auto mutate = std::make_shared<DiagnosticNode>(std::vector<Tensor>{shared});
    mutate->callback = [&] { mul_(x, 2); };
    auto root = attach(mutate);
    EXPECT_THROW(root.backward(), std::runtime_error);
    EXPECT_EQ(mutate->calls, 1);
    EXPECT_EQ(saved->calls, 0);
    EXPECT_FALSE(x.has_grad());
    EXPECT_TRUE(root.has_grad_fn());
    EXPECT_TRUE(is_grad_enabled());
}

TEST(AutogradEngine, UnsupportedMixedLeafPreflightBeforeAnyFormulaOrHook) {
    auto x = leaf();
    x.backward();
    auto bad = ones({2}, DType::Int32);
    bad.requires_grad_(true);
    auto untouched = leaf();
    int hooks = 0;
    x.add_post_accumulate_grad_hook([&] { ++hooks; });
    auto node = std::make_shared<DiagnosticNode>(std::vector<Tensor>{x, bad, untouched});
    auto root = attach(node);
    EXPECT_THROW(root.backward(), std::runtime_error);
    EXPECT_EQ(node->calls, 0);
    EXPECT_EQ(hooks, 0);
    EXPECT_FLOAT_EQ(first(x.grad()), 1);
    EXPECT_FALSE(bad.has_grad());
    EXPECT_FALSE(untouched.has_grad());
}

TEST(AutogradEngine, ReleasedEdgeRejectsBeforeFormulaAndLeavesSiblingValid) {
    auto x = leaf();
    auto shared = mul(x, x);
    auto first_root = scalar_mul(shared, 2);
    auto sibling = scalar_mul(shared, 3);
    first_root.backward();
    auto y = leaf();
    auto node = std::make_shared<DiagnosticNode>(std::vector<Tensor>{shared, y});
    auto invalid = attach(node);
    EXPECT_EQ(node->input_edges()[0].kind, GraphEdge::Kind::Released);
    EXPECT_THROW(invalid.backward(), std::runtime_error);
    EXPECT_EQ(node->calls, 0);
    EXPECT_FALSE(y.has_grad());
    EXPECT_FLOAT_EQ(first(x.grad()), 8);
    sibling.backward();
    EXPECT_FLOAT_EQ(first(x.grad()), 20);
}

TEST(AutogradEngine, FormulaExceptionPreservesLeafStateHistoryAndGradMode) {
    auto x = leaf();
    auto node = std::make_shared<DiagnosticNode>(std::vector<Tensor>{x});
    node->fail = true;
    auto root = attach(node);
    EXPECT_THROW(root.backward(), std::runtime_error);
    EXPECT_FALSE(x.has_grad());
    EXPECT_TRUE(root.has_grad_fn());
    EXPECT_TRUE(is_grad_enabled());
    node->fail = false;
    { NoGradGuard caller; root.backward(); EXPECT_FALSE(is_grad_enabled()); }
    EXPECT_TRUE(is_grad_enabled());
    EXPECT_FLOAT_EQ(first(x.grad()), 1);
}

TEST(AutogradEngine, MalformedGradientRejectsBeforeCommitAndAllowsRetry) {
    auto x = leaf();
    auto node = std::make_shared<DiagnosticNode>(std::vector<Tensor>{x});
    node->malformed = true;
    auto root = attach(node);
    EXPECT_THROW(root.backward(), std::runtime_error);
    EXPECT_FALSE(x.has_grad());
    EXPECT_TRUE(root.has_grad_fn());
    node->malformed = false;
    root.backward();
    EXPECT_FLOAT_EQ(first(x.grad()), 1);
}

TEST(AutogradEngine, OrderedPrefixAndNullSlotsDoNotCreateMissingGradients) {
    auto x = leaf();
    auto frozen = ones({2});
    auto prefix = std::make_shared<DiagnosticNode>(std::vector<Tensor>{x, frozen});
    prefix->prefix = true;
    auto root = attach(prefix);
    root.backward();
    EXPECT_FLOAT_EQ(first(x.grad()), 1);
    EXPECT_FALSE(frozen.has_grad());
    x.zero_grad();
    auto null = std::make_shared<DiagnosticNode>(std::vector<Tensor>{x});
    null->null_slot = true;
    auto absent = attach(null);
    absent.backward();
    EXPECT_FALSE(x.has_grad());
}

TEST(AutogradEngine, HookExceptionKeepsCommittedGradientAndUnreleasedHistory) {
    auto x = leaf();
    auto root = add(x, x);
    x.add_post_accumulate_grad_hook([] { throw std::runtime_error("injected hook failure"); });
    EXPECT_THROW(root.backward(), std::runtime_error);
    EXPECT_FLOAT_EQ(first(x.grad()), 2);
    EXPECT_TRUE(root.has_grad_fn());
    EXPECT_TRUE(is_grad_enabled());
}

TEST(AutogradEngine, GradientsAreIndependentOfSiblingAndForwardBuffers) {
    auto x = leaf(), y = leaf();
    auto root = add(x, y);
    root.backward();
    EXPECT_NE(x.grad().impl()->buffer(), y.grad().impl()->buffer());
    auto gradient = x.grad();
    mul_(gradient, 3);
    EXPECT_FLOAT_EQ(first(y.grad()), 1);
    auto node = std::make_shared<DiagnosticNode>(std::vector<Tensor>{x});
    node->forward_alias = true;
    auto alias_root = attach(node);
    x.zero_grad();
    alias_root.backward();
    EXPECT_NE(x.grad().impl()->buffer(), x.impl()->buffer());
    EXPECT_FALSE(x.grad().requires_grad());
    EXPECT_TRUE(x.grad().is_leaf());
}

TEST(AutogradEngine, Chain4096DiscoversAndReleasesWithoutRecursiveTraversal) {
    auto x = leaf();
    auto root = x;
    std::weak_ptr<Node> first_node;
    for (int i = 0; i < 4096; ++i) {
        root = scalar_add(root, 1);
        if (i == 0) first_node = root.grad_fn();
    }
    root.backward();
    EXPECT_FLOAT_EQ(first(x.grad()), 1);
    EXPECT_TRUE(first_node.expired());
    EXPECT_THROW(root.backward(), std::runtime_error);
}

TEST(AutogradEngine, DirectLeavesRepeatAndSeedEveryCoordinateIncludingScalarAndEmpty) {
    for (const auto& shape : std::vector<std::vector<int64_t>>{{}, {2}, {0}}) {
        auto x = ones(shape);
        x.requires_grad_(true);
        default_autograd_engine().backward(x);
        default_autograd_engine().backward(x, true);
        EXPECT_EQ(x.grad().shape(), shape);
        if (x.numel()) EXPECT_FLOAT_EQ(first(x.grad()), 2);
        EXPECT_FALSE(x.grad().requires_grad());
        EXPECT_TRUE(x.grad().is_leaf());
    }
}

TEST(AutogradEngine, DisjointHostTasksHaveNoSharedExecutionState) {
    auto run = [](float value) {
        auto x = leaf(value);
        auto root = mul(x, x);
        AutogradEngine engine;
        engine.backward(root);
        return first(x.grad());
    };
    auto a = std::async(std::launch::async, run, 2);
    auto b = std::async(std::launch::async, run, 3);
    EXPECT_FLOAT_EQ(a.get(), 4);
    EXPECT_FLOAT_EQ(b.get(), 6);
}

#ifdef WITH_CUDA
TEST(AutogradEngineCuda, SupportedLeafProfilesAndSiblingOwnership) {
    for (auto dtype : {DType::Float32, DType::Float16, DType::BFloat16}) {
        auto x = full({2}, 2, dtype).to(Device(DeviceType::CUDA, 0));
        x.requires_grad_(true);
        auto shared = mul(x, x);
        auto a = scalar_mul(shared, 2), b = scalar_mul(shared, 3);
        a.backward(true);
        b.backward();
        a.backward();
        auto host = cast(x.grad().to(Device{}), DType::Float32);
        EXPECT_FLOAT_EQ(first(host), 28);
        auto independent = ones({2}, dtype).to(x.device());
        independent.requires_grad_(true);
        independent.backward();
        independent.backward();
        auto gradient = cast(independent.grad().to(Device{}), DType::Float32);
        EXPECT_FLOAT_EQ(first(gradient), 2);
    }
}

TEST(AutogradEngineCuda, UnsupportedLeavesRejectBeforeFirstWrite) {
    for (auto dtype : {DType::Float64, DType::Int32, DType::Int64, DType::Bool}) {
        auto x = ones({2}, dtype).to(Device(DeviceType::CUDA, 0));
        x.requires_grad_(true);
        EXPECT_THROW(x.backward(), std::runtime_error);
        EXPECT_FALSE(x.has_grad());
        EXPECT_THROW(x.backward(), std::runtime_error);
        EXPECT_FALSE(x.has_grad());
    }
}
#endif
