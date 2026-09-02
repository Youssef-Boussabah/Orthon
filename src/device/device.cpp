#include <orthon/device/device.h>
#include <stdexcept>

namespace orthon {

Device::Device() : type(DeviceType::CPU), index(0) {}

Device::Device(DeviceType dt, int i) : type(dt), index(i) {
    // Basic invariants only. A positive CUDA ordinal is not checked against
    // the physically present devices here; Device(CUDA, 9999) is a valid
    // value and fails when actually used, same as parsing "cuda:9999".
    switch (dt) {
        case DeviceType::CPU:
            if (i != 0)
                throw std::invalid_argument(
                    "cpu device index must be 0, got " + std::to_string(i));
            break;
        case DeviceType::CUDA:
            if (i < 0)
                throw std::invalid_argument(
                    "cuda device index must be non-negative, got " +
                    std::to_string(i));
            break;
        default:
            throw std::invalid_argument("unknown DeviceType value");
    }
}

bool Device::is_cpu() const{
    return type == DeviceType::CPU;
}

bool Device::is_cuda() const{
    return type == DeviceType::CUDA;
}

bool Device::operator==(const Device& other) const{
    return type == other.type && index == other.index;
}

bool Device::operator!=(const Device& other) const{
    return !(*this == other);
}

Device device_from_string(const std::string& s) {
    if (s == "cpu")
        return Device(DeviceType::CPU, 0);
    if (s == "cuda")
        return Device(DeviceType::CUDA, 0);
    if (s.rfind("cuda:", 0) == 0) {
        // Strict: the suffix must be a non-empty run of decimal digits and
        // nothing else. std::stoi alone would accept "cuda:1x" (as 1) and
        // "cuda:-1", and would report "cuda:" with its own unhelpful message.
        // Leading zeros ("cuda:01") are accepted as ordinary decimal notation.
        const std::string digits = s.substr(5);
        const bool all_digits =
            !digits.empty() &&
            digits.find_first_not_of("0123456789") == std::string::npos;
        // Nine digits keeps the value inside int range for std::stoi.
        if (all_digits && digits.size() <= 9)
            return Device(DeviceType::CUDA, std::stoi(digits));
    }
    throw std::invalid_argument(
        "unknown device '" + s +
        "'; expected 'cpu', 'cuda', or 'cuda:N' (N a non-negative integer)");
}

std::string device_to_string(const Device& d) {
    if (d.is_cpu()) return "cpu";
    return "cuda:" + std::to_string(d.index);
}

}  // namespace orthon
