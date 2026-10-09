#pragma once
#include <orthon/dispatch/operator.h>
namespace orthon {
bool isfinite_all_values(const Tensor&);
std::pair<Tensor, bool> amp_unscale_values(const Tensor&, float);
void amp_copy_values(Tensor&, const Tensor&);
void copy_range_values(Tensor&, const Tensor&, const dispatch::CopyArgs&);
float scalar_read_value(const Tensor&, int64_t, const char*);
void scalar_write_value(const Tensor&, int64_t, float, const char*);
} // namespace orthon
