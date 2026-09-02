#pragma once
#include <orthon/device/device.h>

namespace orthon {

// Host-memory allocation interface. Device memory does not go through it:
// CUDA storage is served by the caching memory pool (memory_pool_allocate),
// which allocates on the tensor's exact device.
class Allocator{
    public:
        virtual ~Allocator() = default;

        virtual void* allocate(size_t n_bytes) = 0;
        virtual void free(void* ptr) = 0;
        virtual Device device() const = 0;

};

Allocator* get_allocator(const Device& device);

}  // namespace orthon
