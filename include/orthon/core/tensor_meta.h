#pragma once
#include <orthon/core/dtype.h>

namespace orthon {

// Logical dtype belongs to Tensor identity, independently of physical bytes.
class TensorMeta {
public:
    explicit TensorMeta(DType dtype) : dtype_(dtype) { (void)dtype_size(dtype); }
    DType dtype() const { return dtype_; }

private:
    DType dtype_;
};

}  // namespace orthon
