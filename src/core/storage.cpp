#include <orthon/core/storage.h>
#include <orthon/core/memory_pool.h>
#include <orthon/memory/allocator.h>

namespace orthon {

Storage::Storage(size_t n_bytes, DType dtype, Device device)
    : ptr(nullptr), n_bytes(n_bytes), dtype(dtype), d(device), alloc(nullptr)
{
    if (device.is_cpu()) {
        alloc = get_allocator(device);
    }
    if (n_bytes > 0) {
        ptr = device.is_cuda()
            ? memory_pool_allocate(n_bytes, device)
            : alloc->allocate(n_bytes);
    }
}

Storage::~Storage(){
    if(ptr != nullptr){
        if (d.is_cuda()) {
            memory_pool_deallocate(ptr, d);
        } else if (alloc != nullptr) {
            alloc->free(ptr);
        }
    }
}

void* Storage::data() const { return ptr; }
size_t Storage::nbytes() const { return n_bytes; }
DType Storage::datatype() const { return dtype; }
Device Storage::device() const { return d; }

}  // namespace orthon
