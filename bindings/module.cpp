#include <pybind11/pybind11.h>
#include <orthon/autograd/grad_mode.h>
#include <orthon/core/memory_pool.h>
#include <orthon/device/device.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace py = pybind11;

using orthon::Device;
using orthon::DeviceType;

// One parser for every device spec string; the strict rules (and their
// error message) live in orthon::device_from_string.
static Device parse_memory_pool_device(const std::string& device) {
    return orthon::device_from_string(device);
}

void bind_dtype(py::module_& m);
void bind_device(py::module_& m);
void bind_tensor(py::module_& m);
void bind_ops(py::module_& m);
void bind_nccl(py::module_& m);

PYBIND11_MODULE(_C, m) {
    m.doc() = "Orthon C++ tensor engine";
    bind_dtype(m);
    bind_device(m);
    bind_tensor(m);
    bind_ops(m);
    bind_nccl(m);

    // Build-time CUDA flag so Python can verify the compiled-in state.
    m.def("cuda_enabled", []() -> bool {
#ifdef WITH_CUDA
        return true;
#else
        return false;
#endif
    });

    // No-op in CPU builds; blocks until all GPU work completes in CUDA builds.
    // Used by benchmarks to get accurate CUDA kernel timing.
    //
    // This is deliberately a CURRENT-DEVICE API, not a tensor-scoped one, and
    // takes no device argument. Unlike a tensor operation - whose authoritative
    // device comes from its operands and is established with a DeviceGuard -
    // "synchronize" is defined in terms of the CUDA context the caller is in.
    m.def("cuda_synchronize", []() {
#ifdef WITH_CUDA
        cudaDeviceSynchronize();
#endif
    });

    m.def("_memory_pool_set_enabled", &orthon::memory_pool_set_enabled);
    m.def("_memory_pool_enabled", &orthon::memory_pool_enabled);
    m.def("_memory_pool_set_max_cached_bytes", &orthon::memory_pool_set_max_cached_bytes);
    m.def("_memory_pool_max_cached_bytes", &orthon::memory_pool_max_cached_bytes);
    m.def("_memory_pool_scratch_allocate", [](size_t n_bytes, const std::string& device) {
        void* ptr = orthon::memory_pool_scratch_allocate(
            n_bytes, parse_memory_pool_device(device));
        return reinterpret_cast<uintptr_t>(ptr);
    }, py::arg("n_bytes"), py::arg("device") = "cpu");
    m.def("_memory_pool_reset_scratch", [](const std::string& device) {
        orthon::memory_pool_reset_scratch(parse_memory_pool_device(device));
    }, py::arg("device") = "cpu");
    m.def("_memory_pool_scratch_stats", [](const std::string& device) {
        orthon::ScratchPoolStats s =
            orthon::memory_pool_scratch_stats(parse_memory_pool_device(device));
        py::dict d;
        d["reserved_bytes"] = s.reserved_bytes;
        d["used_bytes"] = s.used_bytes;
        d["peak_used_bytes"] = s.peak_used_bytes;
        return d;
    }, py::arg("device") = "cpu");
    m.def("_memory_pool_reset_stats", &orthon::memory_pool_reset_stats);
    m.def("_memory_pool_clear_cache", &orthon::memory_pool_clear_cache);
    m.def("_memory_pool_stats", []() {
        orthon::MemoryPoolStats s = orthon::memory_pool_stats();
        py::dict d;
        d["total_allocations"] = s.total_allocations;
        d["raw_allocations"] = s.raw_allocations;
        d["reused_allocations"] = s.reused_allocations;
        d["current_bytes"] = s.current_bytes;
        d["peak_bytes"] = s.peak_bytes;
        d["cached_bytes"] = s.cached_bytes;
        return d;
    });

    m.def("is_grad_enabled", &orthon::is_grad_enabled);
    m.def("set_grad_enabled", [](bool enabled) {
        orthon::set_grad_enabled(enabled);
    }, py::arg("enabled"));
}
