#include <orthon/core/tensor_impl.h>
#include <cstring>
#include <orthon/core/host_copy.h>
namespace orthon {
Tensor make_zeroed_host_tensor(const std::vector<int64_t>& shape, DType dtype) {
    Tensor out(std::make_shared<TensorImpl>(shape, dtype, Device{}));
    if (out.numel() != 0) {
        std::memset(out.data(), 0, static_cast<size_t>(out.numel()) * dtype_size(dtype));
    }
    return out;
}
Tensor make_contiguous_copy(const Tensor& src) {
    const int64_t n = src.numel();
    Tensor out(std::make_shared<TensorImpl>(src.shape(), src.dtype(), src.device()));
    if (n == 0) return out;
    if (src.is_contiguous()) {
        std::memcpy(out.data(), src.data(),
                    static_cast<size_t>(n) * dtype_size(src.dtype()));
    } else {
        const size_t esz = dtype_size(src.dtype());
        char*        dst = static_cast<char*>(out.data());
        const char*  ptr = static_cast<const char*>(src.data());
        const auto& sh = src.shape();
        const auto& st = src.strides();
        const int   nd = static_cast<int>(sh.size());
        std::vector<int64_t> idx(nd, 0);
        for (int64_t elem = 0; elem < n; ++elem) {
            int64_t off = 0;
            for (int d = 0; d < nd; ++d) off += idx[d] * st[d];
            std::memcpy(dst + elem * static_cast<int64_t>(esz),
                        ptr  + off  * static_cast<int64_t>(esz), esz);
            for (int d = nd - 1; d >= 0; --d) {
                if (++idx[d] < sh[d]) break;
                idx[d] = 0;
            }
        }
    }
    return out;
}
}
