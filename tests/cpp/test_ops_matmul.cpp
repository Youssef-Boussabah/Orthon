#include <gtest/gtest.h>
#include <orthon/backends/cpu/kernels.h>
#include <cstring>
#include <orthon/core/tensor.h>
#include <orthon/ops/fill_ops.h>
#include <orthon/ops/shape_ops.h>
#include <orthon/ops/matmul_ops.h>
#include <orthon/planning/matmul_planner.h>
#include <orthon/runtime/runtime.h>
#include <orthon/autograd/matmul_node.h>
#ifdef WITH_CUDA
#include <orthon/backends/cuda/kernels.h>
#include <cuda_runtime_api.h>
#include <orthon/ops/cast_ops.h>
#endif

// The framework's public API lives in namespace orthon. Tests import it wholesale
// so the TEST() bodies read as ordinary API usage; TEST() itself must stay at
// global scope, so this file is not wrapped in the namespace.
using namespace orthon;

TEST(Matmul, Square2x2) {
    // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
    Tensor a = empty({2, 2});
    Tensor b = empty({2, 2});
    float da[] = {1, 2, 3, 4};
    float db[] = {5, 6, 7, 8};
    std::memcpy(a.data(), da, 4 * sizeof(float));
    std::memcpy(b.data(), db, 4 * sizeof(float));
    Tensor c = matmul(a, b);
    ASSERT_EQ(c.shape(), (std::vector<int64_t>{2, 2}));
    const float* p = static_cast<const float*>(c.data());
    EXPECT_FLOAT_EQ(p[0], 19.0f);
    EXPECT_FLOAT_EQ(p[1], 22.0f);
    EXPECT_FLOAT_EQ(p[2], 43.0f);
    EXPECT_FLOAT_EQ(p[3], 50.0f);
}

TEST(Matmul, Rect2x3by3x2) {
    // [[1,2,3],[4,5,6]] @ [[7,8],[9,10],[11,12]] = [[58,64],[139,154]]
    Tensor a = empty({2, 3});
    Tensor b = empty({3, 2});
    float da[] = {1, 2, 3, 4, 5, 6};
    float db[] = {7, 8, 9, 10, 11, 12};
    std::memcpy(a.data(), da, 6 * sizeof(float));
    std::memcpy(b.data(), db, 6 * sizeof(float));
    Tensor c = matmul(a, b);
    ASSERT_EQ(c.shape(), (std::vector<int64_t>{2, 2}));
    const float* p = static_cast<const float*>(c.data());
    EXPECT_FLOAT_EQ(p[0],  58.0f);
    EXPECT_FLOAT_EQ(p[1],  64.0f);
    EXPECT_FLOAT_EQ(p[2], 139.0f);
    EXPECT_FLOAT_EQ(p[3], 154.0f);
}

TEST(Matmul, NonContiguousInput) {
    // transpose([[1,2],[3,4]]) = [[1,3],[2,4]]
    // [[1,3],[2,4]] @ I = [[1,3],[2,4]]
    Tensor base = empty({2, 2});
    float d[] = {1, 2, 3, 4};
    std::memcpy(base.data(), d, 4 * sizeof(float));
    Tensor tr = transpose(base, 0, 1);
    ASSERT_FALSE(tr.is_contiguous());
    Tensor eye = empty({2, 2});
    float de[] = {1, 0, 0, 1};
    std::memcpy(eye.data(), de, 4 * sizeof(float));
    Tensor c = matmul(tr, eye);
    EXPECT_TRUE(c.is_contiguous());
    const float* p = static_cast<const float*>(c.data());
    EXPECT_FLOAT_EQ(p[0], 1.0f);
    EXPECT_FLOAT_EQ(p[1], 3.0f);
    EXPECT_FLOAT_EQ(p[2], 2.0f);
    EXPECT_FLOAT_EQ(p[3], 4.0f);
}

TEST(Matmul, ShapeMismatchThrows) {
    Tensor a = empty({2, 3});
    Tensor b = empty({4, 2});
    EXPECT_THROW(matmul(a, b), std::runtime_error);
}

TEST(Matmul, OutputIsContiguous) {
    Tensor a = ones({3, 3});
    Tensor b = ones({3, 3});
    Tensor c = matmul(a, b);
    EXPECT_TRUE(c.is_contiguous());
}

TEST(MatmulExecution, CpuPlanAndMismatch) {
    for (auto dtype : {DType::Float32,DType::Float64}) for (bool batched : {false,true}) {
        auto a=ones(batched ? std::vector<int64_t>{2,3,4} : std::vector<int64_t>{3,4},dtype);
        auto b=ones(batched ? std::vector<int64_t>{2,4,5} : std::vector<int64_t>{4,5},dtype);
        auto plan=default_matmul_planner().plan(matmul_problem(a,b),default_runtime());
        auto out=backend::cpu::cpu_matmul(a,b,plan);
        EXPECT_FALSE(out.has_grad_fn()); EXPECT_FALSE(out.requires_grad());
        EXPECT_EQ(out.numel(),batched ? 30 : 15);
        if (dtype==DType::Float32) EXPECT_FLOAT_EQ(static_cast<float*>(out.data())[0],4);
        else EXPECT_DOUBLE_EQ(static_cast<double*>(out.data())[0],4);
        const MatmulPlan wrong{plan.problem,MatmulAlgorithm::CudaCustomTiled,plan.preference,
                              plan.work_items,0,false,false,true,false,false};
        EXPECT_THROW(backend::cpu::cpu_matmul(a,b,wrong),std::invalid_argument);
        auto other=ones({2,2},dtype);
        EXPECT_THROW(backend::cpu::cpu_matmul(other,other,plan),std::invalid_argument);
        auto owner=ones(batched ? std::vector<int64_t>{2,4,3} : std::vector<int64_t>{4,3},dtype);
        owner.requires_grad_(true);
        auto view=transpose(owner,-2,-1);
        auto product=matmul(view,b);
        auto node=std::dynamic_pointer_cast<MatMulNode>(product.grad_fn()); ASSERT_TRUE(node);
        EXPECT_EQ(node->inputs[0].impl(),view.impl()); EXPECT_EQ(node->inputs[1].impl(),b.impl());
        owner.impl()->buffer()->bump_version();
        EXPECT_THROW(product.backward(),std::runtime_error);
    }
}
#ifdef WITH_CUDA
namespace {
class CudaMatmulExecution : public ::testing::TestWithParam<std::tuple<DType,int>> {};
TEST_P(CudaMatmulExecution, SelectedPlanExecutesWithoutReselection) {
    int count=0; ASSERT_EQ(cudaGetDeviceCount(&count),cudaSuccess); ASSERT_GT(count,0);
    auto [dtype,rank]=GetParam();
    auto a=cast(ones(rank==2 ? std::vector<int64_t>{3,4} : std::vector<int64_t>{2,3,4}).to({DeviceType::CUDA,0}),dtype);
    auto b=cast(ones(rank==2 ? std::vector<int64_t>{4,5} : std::vector<int64_t>{2,4,5}).to({DeviceType::CUDA,0}),dtype);
    a.requires_grad_(true); b.requires_grad_(true);
    for (auto preference : {MatmulPreference::ForceCustom,MatmulPreference::ForceCublas}) {
        if (dtype==DType::Float32 && preference==MatmulPreference::ForceCublas) continue;
        const auto plan=default_matmul_planner().plan(matmul_problem(a,b),default_runtime(),preference);
        auto out=backend::cuda::cuda_matmul_adapter(a,b,plan);
        EXPECT_FALSE(out.has_grad_fn()); EXPECT_FALSE(out.requires_grad());
        auto host=cast(out.to({}),DType::Float32);
        for (int64_t i=0;i<host.numel();++i) EXPECT_FLOAT_EQ(static_cast<float*>(host.data())[i],4);
        const MatmulPlan wrong{plan.problem,MatmulAlgorithm::CpuReference,plan.preference,
                              plan.work_items,0,false,false,true,false,false};
        EXPECT_THROW(backend::cuda::cuda_matmul_adapter(a,b,wrong),std::invalid_argument);
        auto other=cast(ones({2,2}).to(a.device()),dtype);
        EXPECT_THROW(backend::cuda::cuda_matmul_adapter(other,other,plan),std::invalid_argument);
    }
    auto owner=cast(ones(rank==2 ? std::vector<int64_t>{4,3} : std::vector<int64_t>{2,4,3}).to(a.device()),dtype);
    owner.requires_grad_(true);
    auto view=transpose(owner,-2,-1);
    auto product=matmul(view,b);
    auto node=std::dynamic_pointer_cast<MatMulNode>(product.grad_fn()); ASSERT_TRUE(node);
    EXPECT_EQ(node->inputs[0].impl(),view.impl()); EXPECT_EQ(node->inputs[1].impl(),b.impl());
    owner.impl()->buffer()->bump_version();
    EXPECT_THROW(product.backward(),std::runtime_error);
    // A fabricated plan cannot reinterpret unsupported storage as float32.
    auto unsupported=ones({2,2},DType::Float64).to(a.device());
    const MatmulPlan forged{matmul_problem(unsupported,unsupported),MatmulAlgorithm::CudaCustomTiled,
                           MatmulPreference::ForceCustom,8,0,false,false,true,false,false};
    EXPECT_THROW(backend::cuda::cuda_matmul_adapter(unsupported,unsupported,forged),std::invalid_argument);
}
INSTANTIATE_TEST_SUITE_P(Physical, CudaMatmulExecution, ::testing::Combine(
    ::testing::Values(DType::Float32,DType::Float16,DType::BFloat16),::testing::Values(2,3)));
}
#endif
