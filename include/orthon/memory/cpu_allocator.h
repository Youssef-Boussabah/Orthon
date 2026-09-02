#pragma once
#include <cstddef>
#include <orthon/device/device.h>
#include <orthon/memory/allocator.h>

namespace orthon {

class CPUAllocator : public Allocator{
    public:
        void* allocate(size_t num_bytes) override;
        void free(void* ptr) override;
        Device device() const override;
};

}  // namespace orthon
