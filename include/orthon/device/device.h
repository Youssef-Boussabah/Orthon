#pragma once
#include <string>

namespace orthon {

enum class DeviceType{
    CPU = 0,
    CUDA = 1
};

class Device{
    public:
        Device();
        Device(DeviceType dt, int i);
        bool is_cpu() const;
        bool is_cuda() const;

        bool operator==(const Device& other) const;
        bool operator!=(const Device& other) const;

        DeviceType type;
        int index;
};

// Parse "cpu", "cuda", "cuda:N" → Device.
Device device_from_string(const std::string& s);

// Format Device → "cpu" or "cuda:N".
std::string device_to_string(const Device& d);

}  // namespace orthon
