#include <orthon/core/execution_copy.h>
#include <orthon/core/tensor_impl.h>
#include <orthon/dispatch/dispatcher.h>
#include <orthon/runtime/runtime.h>
namespace orthon {
Tensor execution_operand(const Tensor& t) {
    if (t.is_contiguous()) return t;
    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<TransferKernel>(transfer_request(t, t.device()), default_runtime());
    return kernel.call(t, t.device());
}
Tensor execution_reshape(const Tensor& t, const std::vector<int64_t>& shape) {
    return Tensor(std::make_shared<TensorImpl>(t.impl()->buffer(), Layout(shape, Layout::contiguous_strides(shape), t.impl()->storage_offset()), t.impl()->tensor_meta()));
}
} // namespace orthon
