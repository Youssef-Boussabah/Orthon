#include <orthon/utils/logging.h>
#include <cstdlib>
#include <iostream>

namespace orthon {

bool orthon_debug_device_enabled() {
    const char* value = std::getenv("ORTHON_DEBUG_DEVICE");
    if (value == nullptr) {
        return false;
    }
    const std::string flag(value);
    return !flag.empty() && flag != "0" && flag != "false" && flag != "False";
}

void orthon_log_device_fallback(const std::string& op,
                                  const std::string& reason) {
    if (!orthon_debug_device_enabled()) {
        return;
    }
    std::cerr << "[orthon device] " << op << ": " << reason << std::endl;
}

}  // namespace orthon
