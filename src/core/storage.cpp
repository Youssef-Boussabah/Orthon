#include <orthon/core/storage.h>
#include <orthon/runtime/runtime.h>
#include <orthon/memory/allocator.h>

namespace orthon {

Storage::Storage(size_t n_bytes, Device device)
    : allocation_(default_runtime().allocator(device)->allocate(n_bytes)) {}
Storage::~Storage() = default;
void* Storage::data() const { return allocation_.data(); }
size_t Storage::nbytes() const { return allocation_.nbytes(); }
Device Storage::device() const { return allocation_.device(); }

}  // namespace orthon
