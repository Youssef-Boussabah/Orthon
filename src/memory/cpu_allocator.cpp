#include <orthon/memory/cpu_allocator.h>
#include <orthon/core/memory_pool.h>
#include <orthon/device/device.h>

namespace orthon {

void* CPUAllocator::allocate(size_t num_bytes) {
    return memory_pool_allocate(num_bytes, device());
}

void CPUAllocator::free(void* ptr){
    memory_pool_deallocate(ptr, device());
}

Device CPUAllocator::device() const{
    return Device(DeviceType::CPU, 0);
}

}  // namespace orthon
