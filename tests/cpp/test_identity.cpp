// Namespace and include-tree contract for the native API.
//
// Deliberately does NOT say `using namespace orthon;`. Everything below is
// written the way an external consumer must write it, so the file doubles as a
// header-consumer smoke test: it compiles only if `<orthon/...>` is the include
// spelling and `orthon::` is the qualification.
#include <gtest/gtest.h>

#include <orthon/autograd/grad_mode.h>
#include <orthon/autograd/node.h>
#include <orthon/core/dtype.h>
#include <orthon/core/memory_pool.h>
#include <orthon/core/storage.h>
#include <orthon/core/tensor.h>
#include <orthon/core/tensor_impl.h>
#include <orthon/device/device.h>
#include <orthon/device/device_guard.h>
#include <orthon/memory/allocator.h>
#include <orthon/memory/cpu_allocator.h>
#include <orthon/ops/cpu/binary_ops.h>
#include <orthon/ops/cpu/fill_ops.h>
#include <orthon/ops/cpu/matmul_ops.h>
#include <orthon/ops/cpu/reduce_ops.h>
#include <orthon/ops/cpu/shape_ops.h>
#include <orthon/ops/cpu/unary_ops.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

// ---------------------------------------------------------------------------
// Compile-time guard: the global namespace is free of the framework's names.
//
// These are ordinary global definitions. If any header above still declared a
// type called `Tensor`, or a function called `zeros`, at global scope, the
// corresponding definition here would be a redeclaration - as a different type,
// or as a different KIND of symbol - and this translation unit would fail to
// compile. Nothing here is decorative: deleting the namespace from a public
// header breaks the build at this file.
// ---------------------------------------------------------------------------
struct Tensor      { std::int64_t sentinel; };
struct TensorImpl  { std::int64_t sentinel; };
struct Storage     { std::int64_t sentinel; };
struct Node        { std::int64_t sentinel; };
struct Device      { std::int64_t sentinel; };
struct DeviceType  { std::int64_t sentinel; };
struct DeviceGuard { std::int64_t sentinel; };
struct DType       { std::int64_t sentinel; };
struct Allocator   { std::int64_t sentinel; };
struct CPUAllocator{ std::int64_t sentinel; };
struct GradModeGuard { std::int64_t sentinel; };
struct NoGradGuard { std::int64_t sentinel; };
struct MemoryPoolStats { std::int64_t sentinel; };

// Same idea for free functions: a variable and a function of the same name
// cannot coexist at one scope, so these fail to compile if the ops ever leak
// back out of `orthon`.
int zeros = 0;
int ones = 0;
int full = 0;
int empty = 0;
int add = 0;
int matmul = 0;
int reshape = 0;
int transpose = 0;
int contiguous = 0;
int relu = 0;
int neg = 0;
int memory_pool_allocate = 0;
int memory_pool_stats = 0;
int get_allocator = 0;
int device_from_string = 0;
int device_to_string = 0;
int is_grad_enabled = 0;
int set_grad_enabled = 0;

TEST(Identity, FrameworkTypesAreNotInTheGlobalNamespace) {
    // The global names above are this file's own placeholders, not the
    // framework's - proving the two are distinct types, not aliases.
    EXPECT_FALSE((std::is_same<::Tensor, orthon::Tensor>::value));
    EXPECT_FALSE((std::is_same<::Storage, orthon::Storage>::value));
    EXPECT_FALSE((std::is_same<::Device, orthon::Device>::value));
    EXPECT_FALSE((std::is_same<::Node, orthon::Node>::value));
    EXPECT_FALSE((std::is_same<::DType, orthon::DType>::value));
    EXPECT_FALSE((std::is_same<::Allocator, orthon::Allocator>::value));
    EXPECT_EQ(sizeof(::Tensor), sizeof(std::int64_t));
}

TEST(Identity, PublicTypesResolveUnderOrthon) {
    orthon::Device cpu(orthon::DeviceType::CPU, 0);
    EXPECT_TRUE(cpu.is_cpu());
    EXPECT_FALSE(cpu.is_cuda());
    EXPECT_EQ(orthon::device_to_string(cpu), "cpu");
    EXPECT_TRUE(orthon::device_from_string("cpu").is_cpu());
}

TEST(Identity, FactoriesAndOpsResolveUnderOrthon) {
    orthon::Tensor a = orthon::zeros({2, 3});
    EXPECT_EQ(a.shape(), (std::vector<std::int64_t>{2, 3}));
    EXPECT_EQ(a.numel(), 6);
    EXPECT_EQ(a.dtype(), orthon::DType::Float32);
    EXPECT_TRUE(a.device().is_cpu());

    orthon::Tensor b = orthon::full({2, 3}, 2.0f);
    orthon::Tensor c = orthon::add(a, b);
    const float* p = static_cast<const float*>(c.data());
    for (std::int64_t i = 0; i < c.numel(); ++i) EXPECT_FLOAT_EQ(p[i], 2.0f);

    orthon::Tensor s = orthon::sum(c);
    EXPECT_FLOAT_EQ(static_cast<const float*>(s.data())[0], 12.0f);
}

TEST(Identity, MatmulResolvesUnderOrthon) {
    orthon::Tensor a = orthon::full({2, 3}, 1.0f);
    orthon::Tensor b = orthon::full({3, 4}, 2.0f);
    orthon::Tensor r = orthon::matmul(a, b);
    EXPECT_EQ(r.shape(), (std::vector<std::int64_t>{2, 4}));
    const float* p = static_cast<const float*>(r.data());
    for (std::int64_t i = 0; i < r.numel(); ++i) EXPECT_FLOAT_EQ(p[i], 6.0f);
}

TEST(Identity, MemoryPoolAndGradModeResolveUnderOrthon) {
    const bool grad_was_enabled = orthon::is_grad_enabled();
    orthon::set_grad_enabled(false);
    EXPECT_FALSE(orthon::is_grad_enabled());
    orthon::set_grad_enabled(grad_was_enabled);

    orthon::Device cpu(orthon::DeviceType::CPU, 0);
    void* ptr = orthon::memory_pool_allocate(128, cpu);
    ASSERT_NE(ptr, nullptr);
    orthon::memory_pool_deallocate(ptr, cpu);
}

TEST(Identity, AllocatorServesHostMemoryOnly) {
    orthon::Device cpu(orthon::DeviceType::CPU, 0);
    orthon::Allocator* host = orthon::get_allocator(cpu);
    ASSERT_NE(host, nullptr);
    EXPECT_TRUE(host->device().is_cpu());

    // There is no CUDA allocator: device storage is served by the memory pool
    // on the tensor's exact device, so a CUDA request here is a programming
    // error rather than a fallback.
    orthon::Device gpu(orthon::DeviceType::CUDA, 0);
    EXPECT_THROW(orthon::get_allocator(gpu), std::runtime_error);
}

TEST(Identity, DeviceGuardResolvesUnderOrthon) {
    // CPU construction is a no-op and must stay one, on any build.
    orthon::Device cpu(orthon::DeviceType::CPU, 0);
    orthon::DeviceGuard guard(cpu);
    SUCCEED();
}
