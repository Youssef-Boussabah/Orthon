#include <orthon/memory/allocator.h>
#include <orthon/memory/cpu_allocator.h>
#include <stdexcept>

namespace orthon {

// The Allocator interface serves host memory only. CUDA storage is allocated
// through the caching memory pool with the tensor's exact device, so routing a
// CUDA request here would be a bug rather than a fallback - hence the throw.
Allocator* get_allocator(const Device& device) {
    static CPUAllocator cpu_allocator;
    if (device.is_cpu()) return &cpu_allocator;
    throw std::runtime_error(
        "get_allocator: only CPU memory is served through Allocator; CUDA "
        "storage is allocated by the memory pool on its own device");
}

}  // namespace orthon
