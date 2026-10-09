#include <orthon/core/broadcast.h>
#include <algorithm>
#include <stdexcept>
#include <string>
namespace orthon {
static std::string shape_str(const std::vector<int64_t>& s) {
    std::string r = "[";
    for (size_t i = 0; i < s.size(); ++i) { if (i) r += ", "; r += std::to_string(s[i]); }
    return r + "]";
}
std::vector<int64_t> broadcast_shape(const std::vector<int64_t>& a,
                                     const std::vector<int64_t>& b) {
    const int na = static_cast<int>(a.size());
    const int nb = static_cast<int>(b.size());
    const int nr = std::max(na, nb);
    std::vector<int64_t> out(nr);
    for (int i = 0; i < nr; ++i) {
        int64_t da = (i < nr - na) ? 1 : a[i - (nr - na)];
        int64_t db = (i < nr - nb) ? 1 : b[i - (nr - nb)];
        if (da != db && da != 1 && db != 1)
            throw std::runtime_error(
                "cannot broadcast shapes " + shape_str(a) + " and " + shape_str(b) +
                ": dimension " + std::to_string(i) + " has sizes " +
                std::to_string(da) + " and " + std::to_string(db));
        // Not max(): a size-1 dim stretches to the OTHER size, including 0
        // ([0,3] + [3] is [0,3]). max(0, 1) would fabricate an element and
        // read out of bounds from the empty operand.
        out[i] = (da == 1) ? db : da;
    }
    return out;
}
}
