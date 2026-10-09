#include <orthon/memory/raw_allocator.h>
#include <limits>
#include <new>
#include <stdexcept>

namespace orthon {

size_t checked_allocation_add(size_t a, size_t b) {
    if (b > (std::numeric_limits<size_t>::max)() - a) throw std::bad_alloc();
    return a + b;
}
size_t allocation_size_class(size_t nbytes) {
    return checked_allocation_add(nbytes, 63) & ~size_t(63);
}
namespace {
class SystemRawAllocator final : public RawAllocator {
public:
    void* allocate(size_t capacity, Device device) override {
        if (device.is_cpu()) return cpu_raw_allocate(capacity);
        if (device.is_cuda()) return cuda_raw_allocate(capacity, device);
        throw std::invalid_argument("RawAllocator: unsupported device");
    }
    void deallocate(void* pointer, Device device) noexcept override {
        if (device.is_cpu()) cpu_raw_free(pointer);
        else if (device.is_cuda()) cuda_raw_free(pointer, device);
    }
};
}
std::shared_ptr<RawAllocator> system_raw_allocator() {
    static auto raw = std::make_shared<SystemRawAllocator>();
    return raw;
}

}  // namespace orthon
