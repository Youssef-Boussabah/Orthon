#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <orthon/device/device.h>
#include <orthon/core/dtype.h>
#include <orthon/memory/allocator.h>

namespace orthon {

class Storage{
    public:
        Storage(size_t n_bytes, DType dtype, Device device);
        ~Storage();
        Storage(const Storage&) = delete;
        Storage& operator=(const Storage&) = delete;

        void* data() const;
        size_t nbytes() const;
        DType datatype() const;
        Device device() const;

        // ---- in-place mutation generation ----
        //
        // Counts successful logical in-place mutations of this storage's
        // payload. Views share the Storage and therefore share the counter,
        // which is exactly what saved-tensor validation needs: mutating any
        // alias advances the one generation every alias observes.
        //
        // The counter tracks mutation OPERATIONS, not bytes written: a
        // successful in-place op on a zero-element tensor still bumps.
        // Metadata-only changes (requires_grad, grad, view construction)
        // must not bump. Atomic so a bump from a worker thread (e.g. an
        // async collective enqueue) never races a reader; ordering beyond
        // the count itself is not required, hence relaxed.
        uint64_t version() const {
            return version_.load(std::memory_order_relaxed);
        }
        void bump_version() {
            version_.fetch_add(1, std::memory_order_relaxed);
        }

    private:
        void* ptr;
        size_t n_bytes;
        DType dtype;
        Device d;
        Allocator* alloc;
        std::atomic<uint64_t> version_{0};
};

}  // namespace orthon
