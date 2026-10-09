#include <orthon/dispatch/dispatcher.h>
#include <algorithm>

namespace orthon::dispatch {
namespace {
bool valid_backend(DeviceType backend) {
    return backend == DeviceType::CPU || backend == DeviceType::CUDA;
}
bool overlap(const Constraints& a, const Constraints& b) {
    // Capability::None accepts everything; the only nontrivial predicate is
    // positive, so two predicates can always overlap. Reject conservatively.
    return a.backend == b.backend && a.target_backend == b.target_backend &&
           (a.dtypes & b.dtypes) && (a.layouts & b.layouts) &&
           (a.target_dtypes & b.target_dtypes);
}
bool equal(const Constraints& a, const Constraints& b) {
    return a.backend == b.backend && a.dtypes == b.dtypes &&
           a.layouts == b.layouts && a.target_backend == b.target_backend &&
           a.target_dtypes == b.target_dtypes && a.capability == b.capability;
}
void validate_constraints(const Schema& schema, const Constraints& c) {
    if (!valid_backend(c.backend) || (c.target_backend && !valid_backend(*c.target_backend)) ||
        !c.dtypes || (c.dtypes & ~all_dtypes) || !c.target_dtypes ||
        (c.target_dtypes & ~all_dtypes) || !c.layouts || (c.layouts & ~all_layouts) ||
        (c.capability != Capability::None && c.capability != Capability::CublasBFloat16))
        throw std::invalid_argument("registry: invalid kernel constraints for " + schema.name);
    if ((schema.signature == Signature::Transfer) != c.target_backend.has_value() ||
        (schema.signature != Signature::Cast && c.target_dtypes != all_dtypes) ||
        (c.capability != Capability::None && c.backend != DeviceType::CUDA))
        throw std::invalid_argument("registry: constraints do not match signature for " + schema.name);
}
}  // namespace

void KernelRegistry::require_mutable() const {
    if (frozen_) throw std::logic_error("registry: frozen registry cannot be modified");
}
void KernelRegistry::add_schema(Schema value) {
    require_mutable();
    if (value.name.empty()) throw std::invalid_argument("registry: empty schema name");
    if (static_cast<unsigned>(value.id) > static_cast<unsigned>(OperatorId::ScalarWrite) ||
        static_cast<unsigned>(value.signature) > static_cast<unsigned>(Signature::Shape) ||
        value.arity != signature_arity(value.signature))
        throw std::invalid_argument("registry: invalid schema identity, signature or arity");
    for (const auto& existing : schemas_)
        if (existing.id == value.id || existing.name == value.name)
            throw std::invalid_argument("registry: duplicate operator ID or name: " + value.name);
    schemas_.push_back(std::move(value));
}
const Schema& KernelRegistry::schema(OperatorId id) const {
    for (const auto& value : schemas_) if (value.id == id) return value;
    throw std::out_of_range("registry: unknown operator identity");
}
void KernelRegistry::add_kernel(KernelRegistration value) {
    require_mutable();
    const auto& s = schema(value.op);
    if (s.composite) throw std::invalid_argument("registry: composite operator has no direct kernel: " + s.name);
    if (signature_of(value.kernel) != s.signature)
        throw std::invalid_argument("registry: kernel signature mismatch for " + s.name);
    validate_constraints(s, value.constraints);
    if (value.name.empty() || !std::visit([](const auto& k) { return k.call != nullptr; }, value.kernel))
        throw std::invalid_argument("registry: missing kernel callable or diagnostic name for " + s.name);
    for (const auto& existing : kernels_) {
        if (existing.op != value.op) continue;
        if (equal(existing.constraints, value.constraints))
            throw std::invalid_argument("registry: duplicate kernel registration for " + s.name);
        if (overlap(existing.constraints, value.constraints))
            throw std::invalid_argument("registry: ambiguous overlapping kernels for " + s.name);
    }
    kernels_.push_back(std::move(value));
}
void KernelRegistry::expect_kernel(OperatorId id, Constraints constraints) {
    require_mutable();
    const auto& s = schema(id);
    validate_constraints(s, constraints);
    if (s.composite) throw std::invalid_argument("registry: composite cannot require a kernel");
    expected_.emplace_back(id, constraints);
}
void KernelRegistry::freeze() {
    require_mutable();
    for (const auto& expected : expected_) {
        const auto found = std::find_if(kernels_.begin(), kernels_.end(), [&](const auto& k) {
            return k.op == expected.first && equal(k.constraints, expected.second);
        });
        if (found == kernels_.end())
            throw std::logic_error("registry: expected built-in kernel missing for " + schema(expected.first).name);
    }
    frozen_ = true;
}
}  // namespace orthon::dispatch
