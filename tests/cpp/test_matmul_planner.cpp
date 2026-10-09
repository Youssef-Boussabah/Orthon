#include <gtest/gtest.h>
#include "cuda_test_resources.h"
#include <orthon/planning/matmul_planner.h>
#include <orthon/runtime/runtime.h>
#include <limits>
#include <map>
#include <type_traits>

using namespace orthon;
namespace {
struct Counts { int queried = 0, selected = 0, bound = 0, synced = 0; };
class MetadataResources final : public TestCudaResources {
public:
    MetadataResources(bool capable, Counts& counts) : capable_(capable), counts_(counts) {}
    CudaCapabilities capabilities() const override {
        ++counts_.queried;
        return {8, 9, 130000, capable_, capable_};
    }
    std::unique_ptr<DeviceGuard> select_device() override { ++counts_.selected; return {}; }
    void* cublas_handle() const noexcept override { return nullptr; }
    void bind_compute_stream(void*) override { ++counts_.bound; }
    void synchronize() override { ++counts_.synced; }
private:
    bool capable_;
    Counts& counts_;
};
struct Fixture {
    std::map<int, Counts> counts;
    Runtime runtime{system_raw_allocator(), [this](Device d) {
        return std::make_unique<MetadataResources>(d.index == 0, counts[d.index]);
    }};
};
MatmulProblem problem(DType dtype, int64_t rank = 2, int64_t work = 100, int ordinal = 0) {
    return {{DeviceType::CUDA, ordinal}, dtype, rank, 1, 1, 1, work, false, true};
}
class Boundary : public ::testing::TestWithParam<std::tuple<DType, int64_t, int64_t>> {};
TEST_P(Boundary, ExactInjectedThreshold) {
    Fixture f;
    auto [dtype, rank, work] = GetParam();
    MatmulPlanner planner({100,100,100,100});
    const auto plan = planner.plan(problem(dtype,rank,work),f.runtime);
    EXPECT_EQ(plan.algorithm, work >= 100 ? MatmulAlgorithm::CudaCublas : MatmulAlgorithm::CudaCustomTiled);
    EXPECT_EQ(plan.work_items, work);
    EXPECT_EQ(plan.auto_threshold, 100);
    EXPECT_EQ(plan.threshold_satisfied, work >= 100);
    EXPECT_TRUE(plan.cublas_dtype_eligible);
    EXPECT_TRUE(plan.cublas_capability_eligible);
    EXPECT_TRUE(plan.packs_lhs);
    EXPECT_FALSE(plan.packs_rhs);
}
INSTANTIATE_TEST_SUITE_P(LowPrecision, Boundary, ::testing::Combine(
    ::testing::Values(DType::Float16,DType::BFloat16), ::testing::Values(2,3), ::testing::Values(99,100,101)));
}
TEST(MatmulPlanner, CpuReferenceAndStrides) {
    Runtime runtime;
    for (auto dtype : {DType::Float32,DType::Float64}) for (int rank : {2,3}) {
        const MatmulProblem p{{},dtype,rank,1,3,4,5,false,false};
        const auto plan = default_matmul_planner().plan(p,runtime);
        EXPECT_EQ(plan.algorithm,MatmulAlgorithm::CpuReference);
        EXPECT_EQ(plan.work_items,60);
        EXPECT_FALSE(plan.packs_lhs); EXPECT_FALSE(plan.packs_rhs);
        EXPECT_THROW(default_matmul_planner().plan(p,runtime,MatmulPreference::ForceCublas),std::invalid_argument);
    }
}
TEST(MatmulPlanner, CompatibilityDefaults) {
    Fixture f;
    const auto& planner = default_matmul_planner();
    for (auto rank : {2,3}) {
        EXPECT_EQ(planner.plan(problem(DType::Float32,rank,1),f.runtime).algorithm,MatmulAlgorithm::CudaCustomTiled);
        for (auto dtype : {DType::Float16,DType::BFloat16}) for (int delta : {-1,0,1}) {
            const auto plan = planner.plan(problem(dtype,rank,(16<<20)+delta),f.runtime);
            const bool library = (dtype == DType::Float16 && rank == 2) || delta >= 0;
            EXPECT_EQ(plan.algorithm, library ? MatmulAlgorithm::CudaCublas : MatmulAlgorithm::CudaCustomTiled);
        }
    }
    EXPECT_EQ(planner.plan(problem(DType::Float16,2,1),f.runtime).algorithm,MatmulAlgorithm::CudaCublas);
}
TEST(MatmulPlanner, EachPolicyFamilyIsIndependent) {
    Fixture f; MatmulPlanner planner({10,20,30,40});
    EXPECT_EQ(planner.plan(problem(DType::Float16,2,15),f.runtime).algorithm,MatmulAlgorithm::CudaCublas);
    EXPECT_EQ(planner.plan(problem(DType::Float16,3,15),f.runtime).algorithm,MatmulAlgorithm::CudaCustomTiled);
    EXPECT_EQ(planner.plan(problem(DType::BFloat16,2,35),f.runtime).algorithm,MatmulAlgorithm::CudaCublas);
    EXPECT_EQ(planner.plan(problem(DType::BFloat16,3,35),f.runtime).algorithm,MatmulAlgorithm::CudaCustomTiled);
}
TEST(MatmulPlanner, ExactOrdinalCapabilitiesAndNoLeaseOrSecondCache) {
    Fixture f; MatmulPlanner planner({0,0,0,0});
    for (int i=0;i<5;++i) for (int rank : {2,3}) {
        EXPECT_EQ(planner.plan(problem(DType::BFloat16,rank,1,0),f.runtime).algorithm,MatmulAlgorithm::CudaCublas);
        EXPECT_EQ(planner.plan(problem(DType::BFloat16,rank,1,1),f.runtime).algorithm,MatmulAlgorithm::CudaCustomTiled);
    }
    EXPECT_EQ(f.counts.size(),2u);
    for (const auto& [ordinal,c] : f.counts) {
        EXPECT_EQ(c.queried,1); EXPECT_EQ(c.selected,1); // Context stream creation selects once; planning adds no selections.
         EXPECT_EQ(c.bound,1); EXPECT_EQ(c.synced,0);
    }
}
TEST(MatmulPlanner, ForcedAlgorithmsAndHardEligibility) {
    Fixture f; MatmulPlanner planner({100,100,100,100});
    for (auto dtype : {DType::Float16,DType::BFloat16}) for (int rank : {2,3}) {
        EXPECT_EQ(planner.plan(problem(dtype,rank,1000),f.runtime,MatmulPreference::ForceCustom).algorithm,MatmulAlgorithm::CudaCustomTiled);
        const auto plan=planner.plan(problem(dtype,rank,1),f.runtime,MatmulPreference::ForceCublas);
        EXPECT_EQ(plan.algorithm,MatmulAlgorithm::CudaCublas); EXPECT_FALSE(plan.threshold_satisfied);
        EXPECT_EQ(plan.preference,MatmulPreference::ForceCublas);
    }
    EXPECT_THROW(planner.plan(problem(DType::Float32),f.runtime,MatmulPreference::ForceCublas),std::invalid_argument);
    EXPECT_THROW(planner.plan(problem(DType::BFloat16,3,1,1),f.runtime,MatmulPreference::ForceCublas),std::invalid_argument);
    EXPECT_EQ(planner.plan(problem(DType::BFloat16,2,1,1),f.runtime,MatmulPreference::ForceCustom).algorithm,MatmulAlgorithm::CudaCustomTiled);
}
TEST(MatmulPlanner, RepeatedPlansAreStableImmutableValues) {
    Fixture f; const auto p=problem(DType::BFloat16);
    const auto first=default_matmul_planner().plan(p,f.runtime);
    for (int i=0;i<10;++i) {
        const auto next=default_matmul_planner().plan(p,f.runtime);
        EXPECT_EQ(next.algorithm,first.algorithm); EXPECT_EQ(next.preference,first.preference);
        EXPECT_EQ(next.work_items,first.work_items); EXPECT_EQ(next.auto_threshold,first.auto_threshold);
        EXPECT_EQ(next.cublas_capability_eligible,first.cublas_capability_eligible);
        EXPECT_EQ(next.threshold_satisfied,first.threshold_satisfied);
    }
    static_assert(!std::is_copy_assignable_v<MatmulPlan>);
    static_assert(!std::is_copy_assignable_v<MatmulProblem>);
    static_assert(!std::is_copy_assignable_v<MatmulPlannerPolicy>);
    static_assert(std::is_trivially_destructible_v<MatmulPlan>);
}
TEST(MatmulPlanner, CheckedWorkAndZeroFactors) {
    Fixture f; const auto big=std::numeric_limits<int64_t>::max();
    const MatmulProblem overflow{{},DType::Float32,2,1,big,big,2,true,true};
    EXPECT_THROW(default_matmul_planner().plan(overflow,f.runtime),std::overflow_error);
    const MatmulProblem zero{{},DType::Float32,2,1,big,big,0,true,true};
    EXPECT_EQ(default_matmul_planner().plan(zero,f.runtime).work_items,0u);
    const MatmulProblem batched{{},DType::Float64,3,7,3,5,11,true,true};
    EXPECT_EQ(default_matmul_planner().plan(batched,f.runtime).work_items,1155u);
}
TEST(MatmulPlanner, InvalidMetadataAndExecutionRange) {
    Fixture f;
    const MatmulProblem bad{{},DType::Float32,4,1,1,1,1,true,true};
    EXPECT_THROW(default_matmul_planner().plan(bad,f.runtime),std::invalid_argument);
    EXPECT_THROW(default_matmul_planner().plan(problem(DType::Float64),f.runtime),std::invalid_argument);
    EXPECT_THROW(default_matmul_planner().plan(problem(DType::Float16,2,int64_t{1}<<32),f.runtime,MatmulPreference::ForceCublas),std::invalid_argument);
    EXPECT_THROW(default_matmul_planner().plan(problem(DType::Float16),f.runtime,static_cast<MatmulPreference>(99)),std::invalid_argument);
    const MatmulProblem negative{{},DType::Float32,3,-1,1,1,1,true,true};
    EXPECT_THROW(default_matmul_planner().plan(negative,f.runtime),std::invalid_argument);
}
