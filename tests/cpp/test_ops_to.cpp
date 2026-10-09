#include <gtest/gtest.h>
#include <orthon/core/tensor.h>
#include <orthon/device/device.h>
#include <orthon/ops/fill_ops.h>
#include <orthon/ops/shape_ops.h>
#include <stdexcept>

#ifdef WITH_CUDA
#include <cuda_runtime_api.h>
#endif

// The framework's public API lives in namespace orthon. Tests import it wholesale
// so the TEST() bodies read as ordinary API usage; TEST() itself must stay at
// global scope, so this file is not wrapped in the namespace.
using namespace orthon;

static Device cpu_dev()  { return Device(DeviceType::CPU,  0); }
static Device cuda_dev() { return Device(DeviceType::CUDA, 0); }

#ifdef WITH_CUDA
static int visible_cuda_devices() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess ? n : 0;
}
#endif

TEST(TensorTo, CPU_Contiguous_CopiesValues) {
    Tensor t = full({2, 3}, 7.0f);
    Tensor c = t.to(cpu_dev());
    ASSERT_EQ(c.shape(), t.shape());
    const float* p = static_cast<const float*>(c.data());
    for (int64_t i = 0; i < c.numel(); ++i)
        EXPECT_FLOAT_EQ(p[i], 7.0f);
}

TEST(TensorTo, CPU_ReturnsContiguous) {
    Tensor t = full({2, 3}, 1.0f);
    Tensor c = t.to(cpu_dev());
    EXPECT_TRUE(c.is_contiguous());
}

TEST(TensorTo, CPU_IndependentCopy) {
    Tensor t = full({3}, 1.0f);
    Tensor c = t.to(cpu_dev());
    fill_(t, 99.0f);
    const float* p = static_cast<const float*>(c.data());
    EXPECT_FLOAT_EQ(p[0], 1.0f);
    EXPECT_FLOAT_EQ(p[1], 1.0f);
    EXPECT_FLOAT_EQ(p[2], 1.0f);
}

TEST(TensorTo, NonContiguous_BecomesContiguous) {
    Tensor t  = full({2, 3}, 1.0f);
    Tensor tr = transpose(t, 0, 1);
    ASSERT_FALSE(tr.is_contiguous());
    Tensor c = tr.to(cpu_dev());
    EXPECT_TRUE(c.is_contiguous());
    EXPECT_EQ(c.shape(), tr.shape());
    const float* p = static_cast<const float*>(c.data());
    for (int64_t i = 0; i < c.numel(); ++i)
        EXPECT_FLOAT_EQ(p[i], 1.0f);
}

TEST(TensorTo, PreservesShape) {
    Tensor t = zeros({4, 5, 2});
    Tensor c = t.to(cpu_dev());
    const std::vector<int64_t> expected{4, 5, 2};
    EXPECT_EQ(c.shape(), expected);
}

// Cross-device transfer, per build configuration
//
// The expectation depends on the build, so the tests are split rather than
// written once. In a CPU-only build Tensor::to's cross-device branch is
// compiled out and throws "CUDA is not available in this build". In a CUDA
// build with a device present, CPU -> CUDA transfer must SUCCEED. A single
// unconditional `EXPECT_THROW(t.to(cuda_dev()), ...)` would therefore assert
// the opposite of the contract on exactly the configuration that matters.

#ifndef WITH_CUDA

TEST(TensorTo, CPUOnlyBuildRejectsCudaTarget) {
    Tensor t = zeros({2, 2});
    EXPECT_THROW(t.to(cuda_dev()), std::runtime_error);
}

#else

TEST(TensorTo, CPUToCUDAWorks) {
    if (visible_cuda_devices() < 1) GTEST_SKIP() << "no CUDA device";

    Tensor t = full({2, 3}, 7.0f);
    Tensor c = t.to(cuda_dev());

    // Exact device, not merely "is_cuda".
    EXPECT_TRUE(c.device().is_cuda());
    EXPECT_EQ(c.device().index, 0);
    EXPECT_EQ(device_to_string(c.device()), "cuda:0");

    // Shape and contiguity preserved.
    EXPECT_EQ(c.shape(), t.shape());
    EXPECT_TRUE(c.is_contiguous());

    // Values preserved, verified by a round trip back to the host (the device
    // pointer must not be dereferenced here).
    Tensor back = c.to(cpu_dev());
    ASSERT_EQ(back.shape(), t.shape());
    const float* p = static_cast<const float*>(back.data());
    for (int64_t i = 0; i < back.numel(); ++i)
        EXPECT_FLOAT_EQ(p[i], 7.0f);
}

TEST(TensorTo, CPUToCUDAProducesIndependentStorage) {
    if (visible_cuda_devices() < 1) GTEST_SKIP() << "no CUDA device";

    Tensor t = full({4}, 1.0f);
    Tensor c = t.to(cuda_dev());
    fill_(t, 99.0f);                       // mutate the host source afterwards

    Tensor back = c.to(cpu_dev());
    const float* p = static_cast<const float*>(back.data());
    for (int64_t i = 0; i < back.numel(); ++i)
        EXPECT_FLOAT_EQ(p[i], 1.0f);       // the device copy is unaffected
}

TEST(TensorTo, NonContiguousCPUToCUDALinearises) {
    if (visible_cuda_devices() < 1) GTEST_SKIP() << "no CUDA device";

    Tensor t  = full({2, 3}, 1.0f);
    Tensor tr = transpose(t, 0, 1);
    ASSERT_FALSE(tr.is_contiguous());

    Tensor c = tr.to(cuda_dev());
    EXPECT_TRUE(c.is_contiguous());
    EXPECT_EQ(c.shape(), tr.shape());
    EXPECT_EQ(c.device().index, 0);
}

TEST(TensorTo, InvalidCudaDeviceIndexRaises) {
    if (visible_cuda_devices() < 1) GTEST_SKIP() << "no CUDA device";
    Tensor t = zeros({2, 2});
    EXPECT_THROW(t.to(Device(DeviceType::CUDA, 9999)), std::runtime_error);
}

#endif  // WITH_CUDA

// Strict device-string parsing: exactly "cpu", "cuda", or "cuda:<digits>".

TEST(DeviceFromString, AcceptsCanonicalForms) {
    EXPECT_TRUE(device_from_string("cpu").is_cpu());
    EXPECT_TRUE(device_from_string("cuda").is_cuda());
    EXPECT_EQ(device_from_string("cuda").index, 0);
    EXPECT_EQ(device_from_string("cuda:3").index, 3);
    // Leading zeros are ordinary decimal notation.
    EXPECT_EQ(device_from_string("cuda:01").index, 1);
}

TEST(DeviceFromString, RejectsMalformedStrings) {
    const char* bad[] = {
        "cuda:", "cuda:-1", "cuda:+1", "cuda:1x", "cuda: 1",
        "cuda:01x", "cuda:one", " cuda", "cuda:0 ", "CUDA", "gpu", "",
    };
    for (const char* s : bad)
        EXPECT_THROW(device_from_string(s), std::invalid_argument) << s;
}

// Explicit-constructor validation: basic invariants only. A positive CUDA
// ordinal is not checked against present hardware at construction time.

TEST(DeviceConstructor, AcceptsValidTypeIndexPairs) {
    EXPECT_TRUE(Device(DeviceType::CPU, 0).is_cpu());
    EXPECT_TRUE(Device(DeviceType::CUDA, 0).is_cuda());
    EXPECT_EQ(Device(DeviceType::CUDA, 9999).index, 9999);
}

TEST(DeviceConstructor, RejectsInvalidTypeIndexPairs) {
    EXPECT_THROW(Device(DeviceType::CPU, 1), std::invalid_argument);
    EXPECT_THROW(Device(DeviceType::CPU, -1), std::invalid_argument);
    EXPECT_THROW(Device(DeviceType::CUDA, -1), std::invalid_argument);
    EXPECT_THROW(Device(static_cast<orthon::DeviceType>(42), 0),
                 std::invalid_argument);
}
