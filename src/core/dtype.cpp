#include <orthon/core/dtype.h>
#include <stdexcept>

namespace orthon {

size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float32:  return 4;
        case DType::Float64:  return 8;
        case DType::Float16:  return 2;
        case DType::BFloat16: return 2;
        case DType::Int32:    return 4;
        case DType::Int64:    return 8;
        case DType::UInt8:    return 1;
        case DType::Int8:     return 1;
        case DType::Bool:     return 1;
    }
    throw std::invalid_argument("Unknown DType");
}

bool dtype_is_floating(DType dtype) {
    switch (dtype) {
        case DType::Float32:
        case DType::Float64:
        case DType::Float16:
        case DType::BFloat16:
            return true;
        default:
            return false;
    }
}

}  // namespace orthon
