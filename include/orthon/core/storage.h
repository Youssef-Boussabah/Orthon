#pragma once
#include <cstddef>
#include <cstdint>
#include <orthon/device/device.h>
#include <orthon/memory/allocation.h>

namespace orthon {

// Native compatibility adapter only; Tensor Buffer owns Allocation directly.
class Storage{
    public:
        Storage(size_t n_bytes, Device device);
        ~Storage();
        Storage(const Storage&) = delete;
        Storage& operator=(const Storage&) = delete;

        void* data() const;
        size_t nbytes() const;
        Device device() const;

    private:
        Allocation allocation_;
};

}  // namespace orthon
