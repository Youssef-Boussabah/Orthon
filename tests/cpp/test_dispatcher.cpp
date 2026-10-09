#include <gtest/gtest.h>
#include "cuda_test_resources.h"
#include <orthon/backends/cpu/kernels.h>
#include <orthon/dispatch/dispatcher.h>
#include <orthon/runtime/runtime.h>
#include <orthon/runtime/cuda_context.h>
#include <orthon/core/tensor_impl.h>
#include <orthon/autograd/add_node.h>
#include <orthon/autograd/mul_node.h>
#include <orthon/autograd/grad_mode.h>
#include <orthon/ops/fill_ops.h>
#include <orthon/ops/binary_ops.h>
#include <orthon/ops/cast_ops.h>
#include <orthon/ops/shape_ops.h>
#include <orthon/ops/reduce_ops.h>
#include <orthon/ops/matmul_ops.h>
#include <orthon/planning/matmul_planner.h>
#include <orthon/ops/unary_ops.h>
#include <atomic>
#include <future>
#include <set>
#include <thread>
#ifdef WITH_CUDA
#include <cuda_runtime_api.h>
#endif

using namespace orthon;
using namespace orthon::dispatch;
namespace {
std::atomic<int> selected_calls{0}, other_calls{0};
std::optional<Tensor> failed_output;
Tensor selected(const Tensor& a, const Tensor&, const std::vector<int64_t>& shape) {
    ++selected_calls;
    return full(shape, 17, a.dtype(), a.device());
}
Tensor nonselected(const Tensor&, const Tensor&, const std::vector<int64_t>&) {
    ++other_calls;
    throw std::logic_error("nonselected kernel executed");
}
Tensor failure(const Tensor& a, const Tensor&, const std::vector<int64_t>& shape) {
    ++selected_calls;
    failed_output = empty(shape, a.dtype(), a.device());
    throw std::runtime_error("execution failed");
}
Tensor unary_stub(const Tensor& a) { return a; }
std::shared_ptr<KernelRegistry> isolated() {
    auto r = std::make_shared<KernelRegistry>();
    r->add_schema({OperatorId::Add, "add", Signature::Binary, 2});
    return r;
}
Constraints cpu(DTypeMask dtypes = dtype_bit(DType::Float32), unsigned layouts = all_layouts) {
    return {DeviceType::CPU, dtypes, layouts};
}
KernelRegistration registration(Constraints c, BinaryKernel k = {selected}) {
    return {OperatorId::Add, c, k, "test.add"};
}
DispatchRequest request(Device d = {}, DType dt = DType::Float32,
                        LayoutClass l = LayoutClass::Contiguous) {
    return {OperatorId::Add, d, dt, l, {}, {}};
}
class MetadataResources final : public TestCudaResources {
public:
    explicit MetadataResources(bool supported) : supported_(supported) {}
    CudaCapabilities capabilities() const override {
        return {8, 9, 12000, supported_, supported_};
    }
    std::unique_ptr<DeviceGuard> select_device() override {
        return std::make_unique<DeviceGuard>(Device{});
    }
    void* cublas_handle() const noexcept override { return nullptr; }
    void bind_compute_stream(void*) override {}
    void synchronize() override {}
private:
    bool supported_;
};
}  // namespace

TEST(DispatchSchema, AllCoreIdsHaveUniqueCanonicalSignatures) {
    auto r = build_builtin_registry();
    std::set<std::string> names;
    std::set<OperatorId> ids;
    EXPECT_EQ(r->schemas().size(), 13u + 48u);
    for (const auto& s : r->schemas()) {
        EXPECT_TRUE(names.insert(s.name).second);
        EXPECT_TRUE(ids.insert(s.id).second);
        EXPECT_EQ(&r->schema(s.id), &s);
        EXPECT_EQ(s.arity, signature_arity(s.signature));
    }
    EXPECT_EQ(r->schema(OperatorId::Cast).signature, Signature::Cast);
    EXPECT_EQ(r->schema(OperatorId::To).signature, Signature::Transfer);
    EXPECT_EQ(r->schema(OperatorId::Neg).signature, Signature::Unary);
    EXPECT_EQ(r->schema(OperatorId::Reciprocal).signature, Signature::Unary);
    for (auto id : {OperatorId::Sum, OperatorId::Mean, OperatorId::Max, OperatorId::ArgMax})
        EXPECT_EQ(r->schema(id).signature, Signature::Reduction);
    EXPECT_EQ(r->schema(OperatorId::Matmul).signature, Signature::Matmul);
    for (auto id : {OperatorId::Add, OperatorId::Sub, OperatorId::Mul, OperatorId::Div})
        EXPECT_EQ(r->schema(id).signature, Signature::Binary);
}
TEST(DispatchSchema, DuplicateIdsAndNamesRejectWithoutChangingRegistry) {
    auto r = isolated();
    EXPECT_THROW(r->add_schema({OperatorId::Add, "another", Signature::Binary, 2}), std::invalid_argument);
    EXPECT_THROW(r->add_schema({OperatorId::Mul, "add", Signature::Binary, 2}), std::invalid_argument);
    EXPECT_EQ(r->schemas().size(), 1u);
    EXPECT_NO_THROW(r->add_schema({OperatorId::Mul, "mul", Signature::Binary, 2}));
}
TEST(DispatchSchema, UnknownLookupIsControlled) {
    auto r = build_builtin_registry();
    EXPECT_THROW(r->schema(static_cast<OperatorId>(99)), std::out_of_range);
    auto test = isolated();
    EXPECT_THROW(test->add_schema({static_cast<OperatorId>(99), "unknown", Signature::Binary, 2}),
                 std::invalid_argument);
    EXPECT_THROW(test->add_schema({OperatorId::Mul, "mul", Signature::Unary, 2}), std::invalid_argument);
}
TEST(DispatchSchema, CompositeOperatorsHaveNoRedundantKernel) {
    auto r = build_builtin_registry();
    for (auto id : {OperatorId::Sub, OperatorId::Div}) {
        EXPECT_TRUE(r->schema(id).composite);
        for (const auto& k : r->kernels()) EXPECT_NE(k.op, id);
    }
    auto test = isolated();
    test->add_schema({OperatorId::Sub, "sub", Signature::Binary, 2, true});
    auto k = registration(cpu()); k.op = OperatorId::Sub;
    EXPECT_THROW(test->add_kernel(k), std::invalid_argument);
}
TEST(DispatchRegistry, WrongSignatureInsertionAndLookupReject) {
    auto r = isolated();
    EXPECT_THROW(r->add_kernel({OperatorId::Add, cpu(), UnaryKernel{unary_stub}, "wrong"}), std::invalid_argument);
    EXPECT_TRUE(r->kernels().empty());
    r->add_kernel(registration(cpu())); r->freeze(); Dispatcher d(r); Runtime runtime;
    EXPECT_THROW(d.resolve<UnaryKernel>(request(), runtime), std::logic_error);
    EXPECT_NO_THROW(d.resolve<BinaryKernel>(request(), runtime));
}
TEST(DispatchRegistry, DuplicateExactRegistrationRejectsAndRetryWorks) {
    auto r = isolated(); r->add_kernel(registration(cpu()));
    EXPECT_THROW(r->add_kernel(registration(cpu())), std::invalid_argument);
    EXPECT_EQ(r->kernels().size(), 1u);
    r->add_kernel(registration(cpu(dtype_bit(DType::Float64)), {nonselected}));
    r->freeze(); EXPECT_EQ(r->kernels().size(), 2u);
}
TEST(DispatchRegistry, AmbiguityRejectsInBothInsertionOrders) {
    for (bool reverse : {false, true}) {
        auto r = isolated();
        auto narrow = registration(cpu(dtype_bit(DType::Float32), 1));
        auto broad = registration(cpu(all_dtypes, all_layouts), {nonselected});
        r->add_kernel(reverse ? narrow : broad);
        EXPECT_THROW(r->add_kernel(reverse ? broad : narrow), std::invalid_argument);
        EXPECT_EQ(r->kernels().size(), 1u);
        EXPECT_NO_THROW(r->freeze());
    }
}
TEST(DispatchRegistry, CapabilityConstraintsCannotHideAnOverlap) {
    auto r = isolated();
    Constraints c{DeviceType::CUDA, dtype_bit(DType::BFloat16)};
    r->add_kernel(registration(c)); c.capability = Capability::CublasBFloat16;
    EXPECT_THROW(r->add_kernel(registration(c)), std::invalid_argument);
}
TEST(DispatchRegistry, InvalidConstraintsAndEmptyCallableReject) {
    auto r = isolated();
    for (auto c : {cpu(0), cpu(all_dtypes, 0), cpu(all_dtypes, 8), cpu(1u << 10)})
        EXPECT_THROW(r->add_kernel(registration(c)), std::invalid_argument);
    auto c = cpu(); c.target_backend = DeviceType::CPU;
    EXPECT_THROW(r->add_kernel(registration(c)), std::invalid_argument);
    c = cpu(); c.target_dtypes = dtype_bit(DType::Float32);
    EXPECT_THROW(r->add_kernel(registration(c)), std::invalid_argument);
    c = cpu(); c.capability = Capability::CublasBFloat16;
    EXPECT_THROW(r->add_kernel(registration(c)), std::invalid_argument);
    EXPECT_THROW(r->add_kernel(registration(cpu(), {nullptr})), std::invalid_argument);
    EXPECT_TRUE(r->kernels().empty());
}
TEST(DispatchRegistry, MissingUserImplementationHasMeaningfulUnsupportedError) {
    auto r = isolated(); r->freeze(); Dispatcher d(r); Runtime runtime;
    try { d.resolve<BinaryKernel>(request(), runtime); FAIL(); }
    catch (const UnsupportedOperation& e) {
        EXPECT_NE(std::string(e.what()).find("unsupported backend/dtype/layout"), std::string::npos);
        EXPECT_EQ(std::string(e.what()).find("kernel not found"), std::string::npos);
    }
}
TEST(DispatchRegistry, MissingExpectedBuiltinIsInternalFailureAtFreeze) {
    auto r = isolated(); r->expect_kernel(OperatorId::Add, cpu());
    try { r->freeze(); FAIL(); }
    catch (const std::logic_error& e) {
        EXPECT_NE(std::string(e.what()).find("expected built-in kernel missing"), std::string::npos);
    }
    EXPECT_FALSE(r->frozen());
    r->add_kernel(registration(cpu())); EXPECT_NO_THROW(r->freeze());
}
TEST(DispatchRegistry, FreezeIsRequiredAndAllMutationsThenReject) {
    auto r = isolated(); EXPECT_THROW(Dispatcher d(r), std::logic_error);
    EXPECT_THROW(Dispatcher d(nullptr), std::logic_error);
    r->add_kernel(registration(cpu())); r->freeze();
    EXPECT_THROW(r->add_schema({OperatorId::Mul, "mul", Signature::Binary, 2}), std::logic_error);
    EXPECT_THROW(r->add_kernel(registration(cpu())), std::logic_error);
    EXPECT_THROW(r->expect_kernel(OperatorId::Add, cpu()), std::logic_error);
    EXPECT_THROW(r->freeze(), std::logic_error);
}
TEST(DispatchRegistry, ConstructionIsDeterministicAndIndependentOfProcessDefault) {
    auto a = build_builtin_registry(); auto b = build_builtin_registry();
    EXPECT_NE(a.get(), b.get()); ASSERT_EQ(a->kernels().size(), b->kernels().size());
    for (size_t i = 0; i < a->kernels().size(); ++i) {
        EXPECT_EQ(a->kernels()[i].op, b->kernels()[i].op);
        EXPECT_EQ(a->kernels()[i].name, b->kernels()[i].name);
        EXPECT_EQ(signature_of(a->kernels()[i].kernel), signature_of(b->kernels()[i].kernel));
    }
    auto test = isolated(); test->add_kernel(registration(cpu())); test->freeze();
    EXPECT_NE(&default_dispatcher().registry(), test.get());
    EXPECT_EQ(default_dispatcher().registry().kernels().size(), a->kernels().size());
}
TEST(DispatchRegistry, DispatcherRetainsRegistryThroughCallerDestruction) {
    std::weak_ptr<KernelRegistry> weak;
    {
        auto r = isolated(); r->add_kernel(registration(cpu())); r->freeze(); weak = r;
        { Dispatcher d(r); r.reset(); Runtime runtime;
          EXPECT_FALSE(weak.expired()); EXPECT_NO_THROW(d.resolve<BinaryKernel>(request(), runtime)); }
    }
    EXPECT_TRUE(weak.expired());
}
TEST(DispatchLookup, DTypeAndLayoutConstraintsSelectOnlyTheirKernel) {
    auto r = isolated();
    r->add_kernel(registration(cpu(dtype_bit(DType::Float32), 1)));
    r->add_kernel(registration(cpu(dtype_bit(DType::Float32), 2), {nonselected}));
    r->freeze(); Dispatcher d(r); Runtime runtime;
    EXPECT_EQ(d.resolve<BinaryKernel>(request(), runtime).call, selected);
    EXPECT_EQ(d.resolve<BinaryKernel>(request({}, DType::Float32, LayoutClass::Strided), runtime).call, nonselected);
    EXPECT_THROW(d.resolve<BinaryKernel>(request({}, DType::Float64), runtime), UnsupportedOperation);
    EXPECT_THROW(d.resolve<BinaryKernel>(request({DeviceType::CUDA, 7}), runtime), UnsupportedOperation);
    auto narrow = isolated(); narrow->add_kernel(registration(cpu(dtype_bit(DType::Float32), 1)));
    narrow->freeze(); Dispatcher limited(narrow);
    EXPECT_THROW(limited.resolve<BinaryKernel>(request({}, DType::Float32, LayoutClass::Strided), runtime),
                 UnsupportedOperation);
}
TEST(DispatchLookup, CastTargetDTypeAndTransferTargetBackendAreConstraints) {
    auto registry = std::make_shared<KernelRegistry>();
    registry->add_schema({OperatorId::Cast, "cast", Signature::Cast, 1});
    registry->add_schema({OperatorId::To, "to", Signature::Transfer, 1});
    registry->add_kernel({OperatorId::Cast,
        {DeviceType::CPU, dtype_bit(DType::Float32), all_layouts, {}, dtype_bit(DType::Float64)},
        CastKernel{backend::cpu::cpu_cast}, "test.cast"});
    registry->add_kernel({OperatorId::To, {DeviceType::CPU, all_dtypes, all_layouts, DeviceType::CPU},
                         TransferKernel{backend::cpu::cpu_to}, "test.to"});
    registry->freeze(); Dispatcher d(registry); Runtime runtime; auto a = ones({2});
    EXPECT_NO_THROW(d.resolve<CastKernel>(cast_request(a, DType::Float64), runtime));
    EXPECT_THROW(d.resolve<CastKernel>(cast_request(a, DType::Float16), runtime), UnsupportedOperation);
    EXPECT_NO_THROW(d.resolve<TransferKernel>(transfer_request(a, {}), runtime));
    EXPECT_THROW(d.resolve<TransferKernel>(transfer_request(a, {DeviceType::CUDA,7}), runtime), UnsupportedOperation);
}
TEST(DispatchLookup, MalformedRequestRejectsBeforeExecutingAnything) {
    auto r = isolated(); r->add_kernel(registration(cpu())); r->freeze(); Dispatcher d(r); Runtime runtime;
    auto q = request(); q.target = Device{};
    EXPECT_THROW(d.resolve<BinaryKernel>(q, runtime), std::logic_error);
    q = request(); q.layout = static_cast<LayoutClass>(7);
    EXPECT_THROW(d.resolve<BinaryKernel>(q, runtime), std::logic_error);
    q = request(); q.dtype = static_cast<DType>(77);
    EXPECT_THROW(d.resolve<BinaryKernel>(q, runtime), std::invalid_argument);
}
TEST(DispatchLookup, CapabilityUsesProvidedRuntimeAndExactOrdinalOnly) {
    std::vector<int> queried;
    Runtime runtime(system_raw_allocator(), [&](Device device) {
        queried.push_back(device.index);
        return std::make_unique<MetadataResources>(device.index == 2);
    });
    auto r = isolated();
    Constraints c{DeviceType::CUDA, dtype_bit(DType::BFloat16)};
    c.capability = Capability::CublasBFloat16;
    r->add_kernel(registration(c, {nonselected})); r->freeze(); Dispatcher d(r);
    auto q = request({DeviceType::CUDA, 2}, DType::BFloat16);
    EXPECT_EQ(d.resolve<BinaryKernel>(q, runtime).call, nonselected);
    EXPECT_NO_THROW(d.resolve<BinaryKernel>(q, runtime));
    q.source = {DeviceType::CUDA, 7};
    EXPECT_THROW(d.resolve<BinaryKernel>(q, runtime), UnsupportedOperation);
    EXPECT_EQ(queried, (std::vector<int>{2, 7}));
    // Host-injected metadata tests selection only. No CUDA kernel executes.
}
TEST(DispatchLookup, CPUConstructionAndResolutionNeverCreatesCudaContext) {
    std::atomic<int> resources{0};
    Runtime runtime(system_raw_allocator(), [&](Device) {
        ++resources; return std::make_unique<MetadataResources>(true);
    });
    Dispatcher d(build_builtin_registry());
    EXPECT_NO_THROW(d.resolve<BinaryKernel>(request(), runtime));
    EXPECT_EQ(resources.load(), 0);
}
TEST(DispatchLookup, ConcurrentCpuResolutionHasOneStableIdentity) {
    Dispatcher d(build_builtin_registry()); Runtime runtime;
    const BinaryKernel* canonical = &d.resolve<BinaryKernel>(request(), runtime);
    std::promise<void> gate; auto start = gate.get_future().share();
    std::vector<std::future<const BinaryKernel*>> calls;
    for (int i = 0; i < 16; ++i) calls.push_back(std::async(std::launch::async, [&] {
        start.wait(); const BinaryKernel* found = nullptr;
        for (int j = 0; j < 100; ++j) found = &d.resolve<BinaryKernel>(request(), runtime);
        return found;
    }));
    gate.set_value(); for (auto& call : calls) EXPECT_EQ(call.get(), canonical);
}
TEST(DispatchSemantic, OneResolutionAndOnlySelectedKernelExecutes) {
    selected_calls = 0; other_calls = 0; int resolutions = 0;
    auto r = isolated(); r->add_kernel(registration(cpu()));
    r->add_kernel(registration(cpu(dtype_bit(DType::Float64)), {nonselected})); r->freeze();
    Dispatcher d(r, [&](const KernelRegistration&) { ++resolutions; }); Runtime runtime;
    Tensor a = ones({2,2}); a.requires_grad_(true); Tensor b = ones({2,2});
    auto out = binary_semantic(OperatorId::Add, a, b, d, runtime);
    EXPECT_EQ(resolutions, 1); EXPECT_EQ(selected_calls.load(), 1); EXPECT_EQ(other_calls.load(), 0);
    EXPECT_TRUE(out.requires_grad()); EXPECT_FALSE(out.is_leaf());
    auto node = std::dynamic_pointer_cast<AddNode>(out.grad_fn()); ASSERT_TRUE(node);
    EXPECT_EQ(node->inputs[0].impl(), a.impl()); EXPECT_EQ(node->inputs[1].impl(), b.impl());
    EXPECT_EQ(static_cast<float*>(out.data())[0], 17);
}
TEST(DispatchSemantic, RelationshipAndShapeFailuresExecuteNoKernel) {
    selected_calls = 0; auto r = isolated(); r->add_kernel(registration(cpu())); r->freeze();
    Dispatcher d(r); Runtime runtime; Tensor a = ones({2,2}); a.requires_grad_(true);
    EXPECT_THROW(binary_semantic(OperatorId::Add, a, ones({2,2}, DType::Float64), d, runtime), std::runtime_error);
    EXPECT_THROW(binary_semantic(OperatorId::Add, a, ones({3,3}), d, runtime), std::runtime_error);
    EXPECT_EQ(selected_calls.load(), 0); EXPECT_FALSE(a.has_grad_fn()); EXPECT_FALSE(a.has_grad());
}
TEST(DispatchSemantic, KernelFailureAttachesNoNodeToItsOutputOrInputs) {
    selected_calls = 0; failed_output.reset(); auto r = isolated();
    r->add_kernel(registration(cpu(), {failure})); r->freeze(); Dispatcher d(r); Runtime runtime;
    Tensor a = ones({2,2}); a.requires_grad_(true);
    EXPECT_THROW(binary_semantic(OperatorId::Add, a, a, d, runtime), std::runtime_error);
    EXPECT_EQ(selected_calls.load(), 1); ASSERT_TRUE(failed_output);
    EXPECT_FALSE(failed_output->requires_grad()); EXPECT_FALSE(failed_output->has_grad_fn());
    EXPECT_FALSE(a.has_grad_fn()); EXPECT_FALSE(a.has_grad()); failed_output.reset();
}
TEST(DispatchSemantic, CpuBuiltinAdaptersRemainGraphFreeWithTrackedInputs) {
    Dispatcher d(build_builtin_registry()); Runtime runtime;
    Tensor a = ones({2,2}); a.requires_grad_(true); auto view = transpose(a,0,1);
    for (auto id : {OperatorId::Add, OperatorId::Mul}) {
        auto out = d.resolve<BinaryKernel>(binary_request(id,view,a),runtime).call(view,a,{2,2});
        EXPECT_FALSE(out.requires_grad()); EXPECT_FALSE(out.has_grad_fn()); EXPECT_TRUE(out.is_leaf());
    }
    for (auto id : {OperatorId::Neg, OperatorId::Reciprocal}) {
        auto out = d.resolve<UnaryKernel>(unary_request(id,view),runtime).call(view);
        EXPECT_FALSE(out.requires_grad()); EXPECT_FALSE(out.has_grad_fn());
    }
    for (auto id : {OperatorId::Sum, OperatorId::Mean, OperatorId::Max, OperatorId::ArgMax}) {
        auto out = d.resolve<ReductionKernel>(unary_request(id,view),runtime).call(view, normalize_reduction(view, id == OperatorId::Sum ? ReductionKind::Sum : id == OperatorId::Mean ? ReductionKind::Mean : id == OperatorId::Max ? ReductionKind::Max : ReductionKind::ArgMax));
        EXPECT_FALSE(out.values.requires_grad()); EXPECT_FALSE(out.values.has_grad_fn());
        if (out.indices) EXPECT_FALSE(out.indices->has_grad_fn());
    }
    auto casted = d.resolve<CastKernel>(cast_request(view,DType::Float64),runtime).call(view,DType::Float64);
    EXPECT_FALSE(casted.has_grad_fn()); EXPECT_FALSE(casted.requires_grad());
    auto copied = d.resolve<TransferKernel>(transfer_request(view,{}),runtime).call(view,{});
    EXPECT_FALSE(copied.has_grad_fn()); EXPECT_FALSE(copied.requires_grad());
    auto product = d.resolve<MatmulKernel>(binary_request(OperatorId::Matmul,view,a),runtime).call(view,a,default_matmul_planner().plan(matmul_problem(view,a),runtime));
    EXPECT_FALSE(product.has_grad_fn()); EXPECT_FALSE(product.requires_grad());
}
TEST(DispatchSemantic, SameDeviceCopyAndSameDtypeCastKeepDistinctPolicies) {
    Tensor a = ones({2,2}); a.requires_grad_(true);
    auto copy = a.to(a.device()); auto passthrough = cast(a,a.dtype());
    EXPECT_NE(copy.impl()->buffer(), a.impl()->buffer()); EXPECT_EQ(passthrough.impl(), a.impl());
    EXPECT_TRUE(copy.has_grad_fn()); EXPECT_TRUE(copy.requires_grad());
    EXPECT_FALSE(passthrough.has_grad_fn()); EXPECT_TRUE(passthrough.is_leaf());
    auto view = transpose(a,0,1); auto normalized = cast(view,view.dtype());
    EXPECT_TRUE(normalized.is_contiguous()); EXPECT_NE(normalized.impl()->buffer(), view.impl()->buffer());
    EXPECT_TRUE(normalized.has_grad_fn());
    auto empty_view = a.slice(1,0,0);
    EXPECT_EQ(transfer_request(empty_view,{}).layout, LayoutClass::Empty);
    EXPECT_EQ(empty_view.to({}).numel(),0);
}
TEST(DispatchSemantic, MulSavesOriginalViewsAndDetectsAliasMutation) {
    Tensor a = ones({2,2}); a.requires_grad_(true); auto view = transpose(a,0,1);
    auto out = mul(view,view); auto node = std::dynamic_pointer_cast<MulNode>(out.grad_fn()); ASSERT_TRUE(node);
    EXPECT_EQ(node->inputs[0].impl(), view.impl()); EXPECT_EQ(node->inputs[1].impl(), view.impl());
    mul_(a,2); EXPECT_THROW(out.backward(), std::runtime_error); EXPECT_FALSE(a.has_grad());
}
#ifndef WITH_CUDA
TEST(DispatchCpuOnly, BuiltinRegistryHasNoCudaExecutionAndRejectsTransfers) {
    auto r = build_builtin_registry();
    for (const auto& k : r->kernels()) {
        EXPECT_EQ(k.constraints.backend, DeviceType::CPU);
        if (k.constraints.target_backend) EXPECT_EQ(*k.constraints.target_backend, DeviceType::CPU);
    }
    Dispatcher d(r); Runtime runtime; auto q = request({DeviceType::CUDA,0});
    try { d.resolve<BinaryKernel>(q,runtime); FAIL(); }
    catch(const UnsupportedOperation& e) {
        EXPECT_NE(std::string(e.what()).find("CUDA support not compiled"),std::string::npos);
    }
    Tensor a = ones({2});
    EXPECT_THROW(a.to({DeviceType::CUDA,0}), std::runtime_error);
}
#else
TEST(DispatchCuda, BuiltinRoutesDTypeLayoutsAndTransferDirections) {
    Dispatcher d(build_builtin_registry()); Runtime runtime;
    Tensor host = ones({2,2}); Tensor gpu = host.to({DeviceType::CUDA,0});
    for (auto id : {OperatorId::Add,OperatorId::Mul}) {
        EXPECT_NE(d.resolve<BinaryKernel>(binary_request(id,host,host),runtime).call,
                  d.resolve<BinaryKernel>(binary_request(id,gpu,gpu),runtime).call);
    }
    auto hh = &d.resolve<TransferKernel>(transfer_request(host,{}),runtime);
    auto hg = &d.resolve<TransferKernel>(transfer_request(host,gpu.device()),runtime);
    auto gh = &d.resolve<TransferKernel>(transfer_request(gpu,{}),runtime);
    auto gg = &d.resolve<TransferKernel>(transfer_request(gpu,gpu.device()),runtime);
    EXPECT_NE(hh,hg); EXPECT_NE(hg,gh); EXPECT_NE(gh,gg);
    EXPECT_THROW(d.resolve<CastKernel>(cast_request(gpu,DType::Float64),runtime), UnsupportedOperation);
    EXPECT_THROW(d.resolve<BinaryKernel>(request(gpu.device(),DType::Float64),runtime), UnsupportedOperation);
    auto view = transpose(gpu,0,1);
    EXPECT_NO_THROW(d.resolve<BinaryKernel>(binary_request(OperatorId::Add,view,gpu),runtime));
    auto copied = gpu.to(gpu.device()); EXPECT_NE(copied.impl()->buffer(),gpu.impl()->buffer());
    EXPECT_EQ(copied.device(),gpu.device()); EXPECT_EQ(copied.to({}).numel(),4);
    auto stored = ones({2,3},DType::Int32).to(gpu.device());
    auto empty_view = stored.slice(1,0,0);
    EXPECT_FALSE(empty_view.is_contiguous());
    EXPECT_EQ(transfer_request(empty_view,{}).layout,LayoutClass::Empty);
    EXPECT_EQ(empty_view.to({}).numel(),0);
    EXPECT_EQ(empty_view.to(gpu.device()).numel(),0);
    try { cast(empty_view,DType::Float32); FAIL(); }
    catch(const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("cuda_cast: unsupported dtype pair"),std::string::npos);
    }
}
TEST(DispatchCuda, ConcurrentRealGpuRouteLookupHasStableKernelIdentity) {
    Dispatcher d(build_builtin_registry()); Runtime runtime;
    Tensor t = ones({2}).to({DeviceType::CUDA,0}); auto q = binary_request(OperatorId::Add,t,t);
    const auto* canonical = &d.resolve<BinaryKernel>(q,runtime);
    std::promise<void> gate; auto start=gate.get_future().share();
    std::vector<std::future<const BinaryKernel*>> calls;
    for (int i=0;i<8;++i) calls.push_back(std::async(std::launch::async,[&]{
        start.wait(); const BinaryKernel* found=nullptr;
        for(int j=0;j<100;++j) found=&d.resolve<BinaryKernel>(q,runtime); return found;
    }));
    gate.set_value(); for(auto& call:calls) EXPECT_EQ(call.get(),canonical);
}
TEST(DispatchCuda, ActualExecutionIsGraphFreeUntilSemanticAttachmentAndSavesOriginalViews) {
    int ambient=-1; ASSERT_EQ(cudaGetDevice(&ambient),cudaSuccess);
    Tensor a = ones({2,2}).to({DeviceType::CUDA,0}); a.requires_grad_(true); auto view=transpose(a,0,1);
    Dispatcher d(build_builtin_registry()); Runtime runtime;
    auto raw=d.resolve<BinaryKernel>(binary_request(OperatorId::Mul,view,view),runtime).call(view,view,{2,2});
    EXPECT_FALSE(raw.requires_grad()); EXPECT_FALSE(raw.has_grad_fn());
    for (auto id : {OperatorId::Neg, OperatorId::Reciprocal}) {
        auto result = d.resolve<UnaryKernel>(unary_request(id,view),runtime).call(view);
        EXPECT_FALSE(result.requires_grad()); EXPECT_FALSE(result.has_grad_fn());
    }
    for (auto id : {OperatorId::Sum, OperatorId::Mean, OperatorId::Max, OperatorId::ArgMax}) {
        auto result = d.resolve<ReductionKernel>(unary_request(id,view),runtime).call(view, normalize_reduction(view, id == OperatorId::Sum ? ReductionKind::Sum : id == OperatorId::Mean ? ReductionKind::Mean : id == OperatorId::Max ? ReductionKind::Max : ReductionKind::ArgMax));
        EXPECT_FALSE(result.values.requires_grad()); EXPECT_FALSE(result.values.has_grad_fn());
        if (result.indices) EXPECT_FALSE(result.indices->has_grad_fn());
    }
    auto casted = d.resolve<CastKernel>(cast_request(view,DType::Float16),runtime).call(view,DType::Float16);
    EXPECT_FALSE(casted.requires_grad()); EXPECT_FALSE(casted.has_grad_fn());
    auto copied = d.resolve<TransferKernel>(transfer_request(view,view.device()),runtime).call(view,view.device());
    EXPECT_FALSE(copied.requires_grad()); EXPECT_FALSE(copied.has_grad_fn());
    auto product = d.resolve<MatmulKernel>(binary_request(OperatorId::Matmul,view,a),runtime).call(view,a,default_matmul_planner().plan(matmul_problem(view,a),runtime));
    EXPECT_FALSE(product.requires_grad()); EXPECT_FALSE(product.has_grad_fn());
    auto out=mul(view,view); auto node=std::dynamic_pointer_cast<MulNode>(out.grad_fn()); ASSERT_TRUE(node);
    EXPECT_EQ(node->inputs[0].impl(),view.impl()); EXPECT_EQ(node->inputs[1].impl(),view.impl());
    mul_(a,2); EXPECT_THROW(out.backward(),std::runtime_error); EXPECT_FALSE(a.has_grad());
    int after=-1; ASSERT_EQ(cudaGetDevice(&after),cudaSuccess); EXPECT_EQ(after,ambient);
}
#endif
