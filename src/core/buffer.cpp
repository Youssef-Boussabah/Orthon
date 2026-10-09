#include <orthon/core/buffer.h>
#include <orthon/runtime/runtime.h>
#include <orthon/memory/allocator.h>

namespace orthon {

Buffer::Buffer(size_t nbytes, Device device)
    : allocation_(default_runtime().allocator(device)->allocate(nbytes)) {}
Buffer::~Buffer() = default;
void* Buffer::data() const { return allocation_.data(); }
size_t Buffer::nbytes() const { return allocation_.nbytes(); }
size_t Buffer::capacity() const { return allocation_.capacity(); }
Device Buffer::device() const { return allocation_.device(); }

}  // namespace orthon
