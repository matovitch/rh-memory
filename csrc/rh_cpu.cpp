#include <torch/extension.h>

#include <cmath>
#include <limits>
#include <tuple>

namespace {

inline int64_t normalized_hash(int64_t index, int64_t capacity, int64_t a) {
  int64_t hashed = (a * (index / capacity) + index) % capacity;
  if (hashed < 0) {
    hashed += capacity;
  }
  return hashed;
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> cpu_rh_advance_time(
    const at::Tensor& table_values,
    const at::Tensor& table_gamma,
    const at::Tensor& cutoff_bound_slow_mag,
    const at::Tensor& cutoff_bound_slow_gamma,
    int64_t delta_steps) {
  TORCH_CHECK(table_values.device().is_cpu(), "table_values must be a CPU tensor");
  TORCH_CHECK(table_gamma.device().is_cpu(), "table_gamma must be a CPU tensor");
  TORCH_CHECK(cutoff_bound_slow_mag.device().is_cpu(), "cutoff_bound_slow_mag must be a CPU tensor");
  TORCH_CHECK(cutoff_bound_slow_gamma.device().is_cpu(), "cutoff_bound_slow_gamma must be a CPU tensor");
  TORCH_CHECK(table_values.dim() == 2, "table_values must be 2D");
  TORCH_CHECK(table_gamma.dim() == 2, "table_gamma must be 2D");
  TORCH_CHECK(cutoff_bound_slow_mag.dim() == 1, "cutoff_bound_slow_mag must be 1D");
  TORCH_CHECK(cutoff_bound_slow_gamma.dim() == 1, "cutoff_bound_slow_gamma must be 1D");
  TORCH_CHECK(table_values.sizes() == table_gamma.sizes(), "table_values and table_gamma must have the same shape");
  TORCH_CHECK(cutoff_bound_slow_mag.size(0) == table_values.size(0), "cutoff_bound_slow_mag batch size must match table_values");
  TORCH_CHECK(cutoff_bound_slow_gamma.size(0) == table_values.size(0), "cutoff_bound_slow_gamma batch size must match table_values");
  TORCH_CHECK(table_values.is_contiguous(), "table_values must be contiguous");
  TORCH_CHECK(table_gamma.is_contiguous(), "table_gamma must be contiguous");
  TORCH_CHECK(cutoff_bound_slow_mag.is_contiguous(), "cutoff_bound_slow_mag must be contiguous");
  TORCH_CHECK(cutoff_bound_slow_gamma.is_contiguous(), "cutoff_bound_slow_gamma must be contiguous");

  auto out_values = table_values.clone();
  auto out_gamma = table_gamma.clone();
  auto out_cutoff_bound_slow_mag = cutoff_bound_slow_mag.clone();
  auto out_cutoff_bound_slow_gamma = cutoff_bound_slow_gamma.clone();

  if (delta_steps <= 0) {
    return std::make_tuple(out_values, out_gamma, out_cutoff_bound_slow_mag, out_cutoff_bound_slow_gamma);
  }

  AT_DISPATCH_FLOATING_TYPES(out_values.scalar_type(), "cpu_rh_advance_time", [&] {
    auto* values_ptr = out_values.data_ptr<scalar_t>();
    auto* gamma_ptr = out_gamma.data_ptr<scalar_t>();
    auto* cutoff_bound_slow_mag_ptr = out_cutoff_bound_slow_mag.data_ptr<scalar_t>();
    auto* cutoff_bound_slow_gamma_ptr = out_cutoff_bound_slow_gamma.data_ptr<scalar_t>();

    for (int64_t batch = 0; batch < out_values.size(0); ++batch) {
      const int64_t row_offset = batch * out_values.size(1);
      scalar_t row_min_mag = std::abs(values_ptr[row_offset]);
      scalar_t row_min_gamma = gamma_ptr[row_offset];

      for (int64_t slot = 0; slot < out_values.size(1); ++slot) {
        const int64_t index = row_offset + slot;
        values_ptr[index] *= std::pow(gamma_ptr[index], static_cast<scalar_t>(delta_steps));
        const scalar_t effective = std::abs(values_ptr[index]);
        if (slot == 0 || effective < row_min_mag) {
          row_min_mag = effective;
          row_min_gamma = gamma_ptr[index];
        }
      }

      cutoff_bound_slow_mag_ptr[batch] = row_min_mag;
      cutoff_bound_slow_gamma_ptr[batch] = row_min_gamma;
    }
  });

  return std::make_tuple(out_values, out_gamma, out_cutoff_bound_slow_mag, out_cutoff_bound_slow_gamma);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> cpu_rh_write(
    const at::Tensor& table_values,
    const at::Tensor& table_dib,
    const at::Tensor& table_gamma,
    const at::Tensor& incoming_values,
    const at::Tensor& incoming_indices,
    const at::Tensor& incoming_gammas,
    int64_t capacity,
    int64_t a) {
  TORCH_CHECK(table_values.device().is_cpu(), "table_values must be a CPU tensor");
  TORCH_CHECK(table_dib.device().is_cpu(), "table_dib must be a CPU tensor");
  TORCH_CHECK(table_gamma.device().is_cpu(), "table_gamma must be a CPU tensor");
  TORCH_CHECK(incoming_values.device().is_cpu(), "incoming_values must be a CPU tensor");
  TORCH_CHECK(incoming_indices.device().is_cpu(), "incoming_indices must be a CPU tensor");
  TORCH_CHECK(incoming_gammas.device().is_cpu(), "incoming_gammas must be a CPU tensor");
  TORCH_CHECK(table_values.sizes() == table_dib.sizes(), "table_values and table_dib must have the same shape");
  TORCH_CHECK(table_values.sizes() == table_gamma.sizes(), "table_values and table_gamma must have the same shape");
  TORCH_CHECK(incoming_values.dim() == 1, "incoming_values must be 1D");
  TORCH_CHECK(incoming_indices.dim() == 1, "incoming_indices must be 1D");
  TORCH_CHECK(incoming_gammas.dim() == 1, "incoming_gammas must be 1D");
  TORCH_CHECK(incoming_values.sizes() == incoming_indices.sizes(), "incoming_values and incoming_indices must have the same shape");
  TORCH_CHECK(incoming_values.sizes() == incoming_gammas.sizes(), "incoming_values and incoming_gammas must have the same shape");
  TORCH_CHECK(capacity > 0, "capacity must be positive");
  TORCH_CHECK(a != 0, "a must be non-zero");

  TORCH_CHECK(table_values.is_contiguous(), "table_values must be contiguous");
  TORCH_CHECK(table_dib.is_contiguous(), "table_dib must be contiguous");
  TORCH_CHECK(table_gamma.is_contiguous(), "table_gamma must be contiguous");
  TORCH_CHECK(incoming_values.is_contiguous(), "incoming_values must be contiguous");
  TORCH_CHECK(incoming_indices.is_contiguous(), "incoming_indices must be contiguous");
  TORCH_CHECK(incoming_gammas.is_contiguous(), "incoming_gammas must be contiguous");

  TORCH_CHECK(table_values.dim() == 1, "table_values must be 1D");
  TORCH_CHECK(table_values.size(0) == capacity, "table_values length must match capacity");

  auto out_values = table_values.clone();
  auto out_dib = table_dib.clone();
  auto out_gamma = table_gamma.clone();

  AT_DISPATCH_FLOATING_TYPES(out_values.scalar_type(), "cpu_rh_write", [&] {
    auto* values_ptr = out_values.data_ptr<scalar_t>();
    auto* gamma_ptr = out_gamma.data_ptr<scalar_t>();
    auto* incoming_values_ptr = incoming_values.data_ptr<scalar_t>();
    auto* incoming_gammas_ptr = incoming_gammas.data_ptr<scalar_t>();
    auto* dib_ptr = out_dib.data_ptr<int64_t>();
    auto* incoming_indices_ptr = incoming_indices.data_ptr<int64_t>();

    const auto incoming_count = incoming_values.size(0);
    for (int64_t item = 0; item < incoming_count; ++item) {
      scalar_t current_value = incoming_values_ptr[item];
      
      if (std::abs(current_value) == 0.0) {
          break;
      }

      scalar_t current_gamma = incoming_gammas_ptr[item];
      const int64_t current_index = incoming_indices_ptr[item];
      int64_t current_dib = 0;
      int64_t current_bucket = normalized_hash(current_index, capacity, a);

      for (int64_t probe = 0; probe < capacity; ++probe) {
        const int64_t slot = (current_bucket + probe) % capacity;
        const scalar_t current_eff = std::abs(current_value);
        const scalar_t slot_eff = std::abs(values_ptr[slot]);
        const bool current_wins = current_eff > slot_eff || (current_eff == slot_eff && current_dib > dib_ptr[slot]);

        if (!current_wins) {
          continue;
        }

        const scalar_t displaced_value = values_ptr[slot];
        const scalar_t displaced_gamma = gamma_ptr[slot];
        const int64_t displaced_dib = dib_ptr[slot];

        values_ptr[slot] = current_value;
        gamma_ptr[slot] = current_gamma;
        dib_ptr[slot] = current_dib;

        current_value = displaced_value;
        current_gamma = displaced_gamma;
        current_dib = displaced_dib + 1;
      }
    }
  });

  return std::make_tuple(out_values, out_dib, out_gamma);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> cpu_rh_write_batched(
    const at::Tensor& table_values,
    const at::Tensor& table_dib,
    const at::Tensor& table_gamma,
    const at::Tensor& incoming_values,
    const at::Tensor& incoming_indices,
    const at::Tensor& incoming_gammas,
    int64_t capacity,
    int64_t a) {
  TORCH_CHECK(table_values.dim() == 2, "table_values must be 2D");
  TORCH_CHECK(table_dib.dim() == 2, "table_dib must be 2D");
  TORCH_CHECK(table_gamma.dim() == 2, "table_gamma must be 2D");
  TORCH_CHECK(incoming_values.dim() == 2, "incoming_values must be 2D");
  TORCH_CHECK(incoming_indices.dim() == 2, "incoming_indices must be 2D");
  TORCH_CHECK(incoming_gammas.dim() == 2, "incoming_gammas must be 2D");
  TORCH_CHECK(table_values.sizes() == table_dib.sizes(), "table_values and table_dib must have the same shape");
  TORCH_CHECK(table_values.sizes() == table_gamma.sizes(), "table_values and table_gamma must have the same shape");
  TORCH_CHECK(incoming_values.sizes() == incoming_indices.sizes(), "incoming_values and incoming_indices must have the same shape");
  TORCH_CHECK(incoming_values.sizes() == incoming_gammas.sizes(), "incoming_values and incoming_gammas must have the same shape");
  TORCH_CHECK(table_values.size(1) == capacity, "table_values second dimension must match capacity");
  TORCH_CHECK(incoming_values.size(0) == table_values.size(0), "batch size must match between table and incoming values");

  auto out_values = table_values.clone();
  auto out_dib = table_dib.clone();
  auto out_gamma = table_gamma.clone();

  for (int64_t batch = 0; batch < table_values.size(0); ++batch) {
    auto batch_result = cpu_rh_write(
        out_values.select(0, batch),
        out_dib.select(0, batch),
        out_gamma.select(0, batch),
        incoming_values.select(0, batch),
        incoming_indices.select(0, batch),
        incoming_gammas.select(0, batch),
        capacity,
        a);

    out_values.select(0, batch).copy_(std::get<0>(batch_result));
    out_dib.select(0, batch).copy_(std::get<1>(batch_result));
    out_gamma.select(0, batch).copy_(std::get<2>(batch_result));
  }

  return std::make_tuple(out_values, out_dib, out_gamma);
}

TORCH_LIBRARY(rh_memory, m) {
  m.def("cpu_rh_write(Tensor table_values, Tensor table_dib, Tensor table_gamma, Tensor incoming_values, Tensor incoming_indices, Tensor incoming_gammas, int capacity, int a=1) -> (Tensor, Tensor, Tensor)");
  m.def("cpu_rh_write_batched(Tensor table_values, Tensor table_dib, Tensor table_gamma, Tensor incoming_values, Tensor incoming_indices, Tensor incoming_gammas, int capacity, int a=1) -> (Tensor, Tensor, Tensor)");
  m.def("cpu_rh_advance_time(Tensor table_values, Tensor table_gamma, Tensor cutoff_bound_slow_mag, Tensor cutoff_bound_slow_gamma, int delta_steps) -> (Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(rh_memory, CPU, m) {
  m.impl("cpu_rh_write", cpu_rh_write);
  m.impl("cpu_rh_write_batched", cpu_rh_write_batched);
  m.impl("cpu_rh_advance_time", cpu_rh_advance_time);
}