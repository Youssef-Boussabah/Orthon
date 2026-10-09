#pragma once
#include <orthon/dispatch/operator.h>
#include <functional>
#include <memory>
#include <stdexcept>

namespace orthon {
class Runtime;
namespace dispatch {

enum class Capability { None, CublasBFloat16 };
struct Constraints {
    DeviceType backend;
    DTypeMask dtypes;
    unsigned layouts = all_layouts;
    std::optional<DeviceType> target_backend;
    DTypeMask target_dtypes = all_dtypes;
    Capability capability = Capability::None;
};
struct KernelRegistration {
    OperatorId op;
    Constraints constraints;
    Kernel kernel;
    std::string name;
};

class UnsupportedOperation : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class KernelRegistry {
public:
    KernelRegistry() = default;
    KernelRegistry(const KernelRegistry&) = delete;
    KernelRegistry& operator=(const KernelRegistry&) = delete;
    void add_schema(Schema schema);
    const Schema& schema(OperatorId id) const;
    const std::vector<Schema>& schemas() const noexcept { return schemas_; }
    void add_kernel(KernelRegistration registration);
    // An expected built-in absent at freeze is corruption, not user rejection.
    void expect_kernel(OperatorId id, Constraints constraints);
    void freeze();
    bool frozen() const noexcept { return frozen_; }
    const std::vector<KernelRegistration>& kernels() const noexcept { return kernels_; }
private:
    friend class Dispatcher;
    void require_mutable() const;
    std::vector<Schema> schemas_;
    std::vector<KernelRegistration> kernels_;
    std::vector<std::pair<OperatorId, Constraints>> expected_;
    bool frozen_ = false;
};

class Dispatcher {
public:
    // Observer is an isolated native diagnostic seam, absent in production.
    using Observer = std::function<void(const KernelRegistration&)>;
    explicit Dispatcher(std::shared_ptr<const KernelRegistry> registry,
                        Observer observer = {});
    const Schema& schema(OperatorId id) const { return registry_->schema(id); }
    const KernelRegistry& registry() const noexcept { return *registry_; }
    template<typename K>
    const K& resolve(const DispatchRequest& request, Runtime& runtime) const {
        if (schema(request.op).signature != K::signature)
            throw std::logic_error("dispatch: kernel signature mismatch for " + schema(request.op).name);
        const auto& selected = select(request, runtime);
        const auto* kernel = std::get_if<K>(&selected.kernel);
        if (!kernel)
            throw std::logic_error("dispatch: corrupt kernel signature for " + schema(request.op).name);
        if (observer_) observer_(selected);
        return *kernel;
    }
    // Compatibility preflight for reductions: legacy CPU dtype rejection
    // precedes dimension errors, CUDA rejection follows them. This checks only
    // declared dtype support; it neither resolves nor executes a route.
    void check_dtype(const DispatchRequest& request) const;
private:
    const KernelRegistration& select(const DispatchRequest&, Runtime&) const;
    std::shared_ptr<const KernelRegistry> registry_;
    Observer observer_;
};

std::shared_ptr<const KernelRegistry> build_builtin_registry();
const Dispatcher& default_dispatcher();
std::string unsupported_diagnostic(const Schema&, const DispatchRequest&);

}  // namespace dispatch
}  // namespace orthon
