#include <gtest/gtest.h>
#include <orthon/device/device_guard.h>
#include <stdexcept>

// The framework's public API lives in namespace orthon. Tests import it wholesale
// so the TEST() bodies read as ordinary API usage; TEST() itself must stay at
// global scope, so this file is not wrapped in the namespace.
using namespace orthon;

#ifdef WITH_CUDA
#include <cuda_runtime_api.h>

namespace {
int current_device() {
    int d = -1;
    EXPECT_EQ(cudaGetDevice(&d), cudaSuccess);
    return d;
}
int visible_devices() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess ? n : 0;
}
}  // namespace
#endif

// CPU: a guard on a CPU Device is a no-op in every build
TEST(DeviceGuard, CpuDeviceIsANoOp) {
    EXPECT_NO_THROW({ DeviceGuard guard(Device{}); });
}

#ifndef WITH_CUDA

TEST(DeviceGuard, CpuOnlyBuildRejectsACudaDevice) {
    EXPECT_THROW(DeviceGuard guard(Device(DeviceType::CUDA, 0)), std::runtime_error);
}

#else

TEST(DeviceGuard, CpuGuardLeavesTheCurrentCudaDeviceUntouched) {
    if (visible_devices() < 1) GTEST_SKIP() << "no CUDA device";
    const int before = current_device();
    { DeviceGuard guard(Device{}); }
    EXPECT_EQ(current_device(), before);
}

TEST(DeviceGuard, SelectsTheRequestedDeviceAndRestores) {
    if (visible_devices() < 1) GTEST_SKIP() << "no CUDA device";
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    int inside = -1;
    {
        DeviceGuard guard(Device(DeviceType::CUDA, 0));
        inside = current_device();
    }
    EXPECT_EQ(inside, 0);
    EXPECT_EQ(current_device(), 0);
}

TEST(DeviceGuard, InvalidDeviceIndexRaises) {
    if (visible_devices() < 1) GTEST_SKIP() << "no CUDA device";
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    EXPECT_THROW(DeviceGuard guard(Device(DeviceType::CUDA, 9999)), std::runtime_error);
    // A failed construction must not have disturbed the current device.
    EXPECT_EQ(current_device(), 0);
}

TEST(DeviceGuard, ExceptionUnwindingRestoresSameDevice) {
    if (visible_devices() < 1) GTEST_SKIP() << "no CUDA device";
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    try {
        DeviceGuard guard(Device(DeviceType::CUDA, 0));
        throw std::runtime_error("boom");
    } catch (const std::runtime_error&) {
    }
    EXPECT_EQ(current_device(), 0);
}

// Multi-GPU: these are the cases that actually prove restore-on-exit
TEST(DeviceGuard, SelectsASecondDeviceAndRestoresTheFirst) {
    if (visible_devices() < 2) GTEST_SKIP() << "needs >=2 CUDA devices";
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    int inside = -1;
    {
        DeviceGuard guard(Device(DeviceType::CUDA, 1));
        inside = current_device();
    }
    EXPECT_EQ(inside, 1);
    EXPECT_EQ(current_device(), 0);
}

TEST(DeviceGuard, NestedGuardsUnwindInOrder) {
    if (visible_devices() < 2) GTEST_SKIP() << "needs >=2 CUDA devices";
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    int lvl1 = -1, lvl2 = -1, after_inner = -1;
    {
        DeviceGuard outer(Device(DeviceType::CUDA, 1));
        lvl1 = current_device();
        {
            DeviceGuard inner(Device(DeviceType::CUDA, 0));
            lvl2 = current_device();
        }
        after_inner = current_device();
    }
    EXPECT_EQ(lvl1, 1);          // 0 -> 1
    EXPECT_EQ(lvl2, 0);          //      -> 0
    EXPECT_EQ(after_inner, 1);   //      -> 1
    EXPECT_EQ(current_device(), 0);
}

TEST(DeviceGuard, ExceptionUnwindingRestoresAcrossDevices) {
    if (visible_devices() < 2) GTEST_SKIP() << "needs >=2 CUDA devices";
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    try {
        DeviceGuard guard(Device(DeviceType::CUDA, 1));
        EXPECT_EQ(current_device(), 1);
        throw std::runtime_error("boom");
    } catch (const std::runtime_error&) {
    }
    EXPECT_EQ(current_device(), 0);
}

#endif  // WITH_CUDA
