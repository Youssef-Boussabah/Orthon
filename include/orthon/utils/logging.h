#pragma once
#include <string>

namespace orthon {

bool orthon_debug_device_enabled();
void orthon_log_device_fallback(const std::string& op,
                                  const std::string& reason);

}  // namespace orthon
