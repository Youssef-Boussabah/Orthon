#pragma once
#include <orthon/device/device.h>

namespace orthon {

// RAII CUDA device-context guard.
//
// A CUDA operation must not depend on whichever device happened to be current
// when it was called. Constructing a DeviceGuard from the operation's
// authoritative tensor Device makes that device current for the guard's scope
// and restores the caller's previous device on the way out - including when the
// scope is left by an exception.
//
//     {
//         DeviceGuard guard(t.device());   // cuda:1 becomes current
//         ...allocations, memcpys, kernel launches, cuBLAS calls...
//     }                                    // caller's device is current again
//
// Cheap and re-entrant: a guard whose target is already current records nothing
// and restores nothing, so nesting guards on the same device costs one
// cudaGetDevice. Constructing from a CPU Device is a no-op, so a guard may be
// placed unconditionally at a dispatch point that serves both devices.
//
// This header deliberately does not include any CUDA header, so CPU-only
// consumers never acquire a CUDA dependency; the runtime calls live in
// device_guard.cpp.
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
