#include <orthon/device/device_guard.h>
#include <orthon/utils/logging.h>
#include <stdexcept>
#include <string>

#ifdef WITH_CUDA
#include <cuda_runtime_api.h>
#endif

namespace orthon {

DeviceGuard::DeviceGuard(const Device& device)
    : changed_(false), previous_device_(0) {
    if (!device.is_cuda()) return;   // CPU: nothing to select.

#ifdef WITH_CUDA
    int current = 0;
    cudaError_t err = cudaGetDevice(&current);
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("DeviceGuard: cudaGetDevice failed: ") +
            cudaGetErrorString(err));

    // Already current: record nothing so the destructor stays a no-op. This is
    // what makes nesting guards on one device essentially free.
    if (current == device.index) return;

    err = cudaSetDevice(device.index);
    if (err != cudaSuccess)
        throw std::runtime_error(
            "DeviceGuard: cudaSetDevice(" + std::to_string(device.index) +
            ") failed: " + cudaGetErrorString(err));

    previous_device_ = current;
    changed_ = true;
#else
    throw std::runtime_error(
        "DeviceGuard: CUDA device " + std::to_string(device.index) +
        " requested but CUDA is not available in this build "
        "(recompile with -DWITH_CUDA=ON)");
#endif
}

DeviceGuard::~DeviceGuard() noexcept {
    if (!changed_) return;

#ifdef WITH_CUDA
    // Destructors run during stack unwinding, so this cannot throw. The project
    // already has a device-diagnostic channel for exactly this kind of report;
    // it stays silent unless ORTHON_DEBUG_DEVICE is set.
    cudaError_t err = cudaSetDevice(previous_device_);
    if (err != cudaSuccess)
        orthon_log_device_fallback(
            "DeviceGuard",
            "failed to restore CUDA device " + std::to_string(previous_device_) +
                ": " + cudaGetErrorString(err));
#endif
}

}  // namespace orthon
