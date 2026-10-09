#pragma once
#include <orthon/device/device.h>

namespace orthon {

// Select the operation's exact CUDA device and restore the caller's device,
// including on exceptions. CPU and already-current targets need no restore;
// nested guards on the same GPU are cheap. CUDA calls live in the source
// file so this header remains usable by CPU-only consumers.
class DeviceGuard {
public:
    // Selects `device` if it is a CUDA device that is not already current.
    // Throws std::runtime_error if the CUDA runtime rejects the query or the
    // selection, or if a CUDA device is requested from a CPU-only build.
    explicit DeviceGuard(const Device& device);

    // Restores the previously current CUDA device. Never throws: a failure here
    // is reported through the project's existing device-diagnostic channel
    // (ORTHON_DEBUG_DEVICE) rather than terminating during stack unwinding.
    ~DeviceGuard() noexcept;

    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;
    DeviceGuard(DeviceGuard&&) = delete;
    DeviceGuard& operator=(DeviceGuard&&) = delete;

private:
    bool changed_;
    int  previous_device_;
};

}  // namespace orthon
