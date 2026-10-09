#pragma once
#include <cstdint>

namespace orthon {
// One selector for CPU traversal, CUDA threads and CUDA partial/tree merges.
// Negative indices mark padding, so an all -Inf input has a real first winner.
template<typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
inline bool extrema_candidate_wins(T candidate, int64_t candidate_index,
                                   T current, int64_t current_index) {
    if (candidate_index < 0) return false;
    if (current_index < 0) return true;
    const bool candidate_nan = candidate != candidate;
    const bool current_nan = current != current;
    if (candidate_nan != current_nan) return candidate_nan;
    if (candidate_nan) return candidate_index < current_index;
    if (candidate > current) return true;
    return candidate == current && candidate_index < current_index;
}
} // namespace orthon
