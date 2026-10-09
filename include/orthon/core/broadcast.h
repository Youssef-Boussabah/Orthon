#pragma once
#include <cstdint>
#include <vector>
namespace orthon {
std::vector<int64_t> broadcast_shape(const std::vector<int64_t>& a,
                                     const std::vector<int64_t>& b) ;
} // namespace orthon
