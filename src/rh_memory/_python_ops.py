"""Pure Python RH write helpers."""

from __future__ import annotations

import torch


def _python_rh_write_single(
	table_values: torch.Tensor,
	table_dib: torch.Tensor,
	table_gamma: torch.Tensor,
	incoming_values: torch.Tensor,
	incoming_indices: torch.Tensor,
	incoming_gammas: torch.Tensor,
	cutoff_bound_slow_mag: torch.Tensor,
	cutoff_bound_slow_gamma: torch.Tensor,
	capacity: int,
	a: int = 1,
	b: int = 0,
):
	"""Apply one RH write batch to a single table.

	Contract: the first batch that reaches an empty table is expected to contain at
	least `capacity` items so the table becomes fully occupied. After that, later
		write batches may be smaller; the steady-state branch assumes the table is full.
		Writes rank items by raw absolute magnitude; gamma is stored so `advance_time`
		can decay the table and cutoff bound over elapsed time.
	"""
	if table_values.dim() != 1:
		raise ValueError("table_values must be 1D")
	if table_dib.dim() != 1:
		raise ValueError("table_dib must be 1D")
	if table_gamma.dim() != 1:
		raise ValueError("table_gamma must be 1D")

	out_values = table_values.clone()
	out_dib = table_dib.clone()
	out_gamma = table_gamma.clone()
	last_inserted_value: torch.Tensor | None = None
	last_inserted_gamma: torch.Tensor | None = None

	for value, index, gamma in zip(incoming_values, incoming_indices, incoming_gammas):
		current_value = value
		current_gamma = gamma
		current_dib = 0
		current_bucket = int((a * int(index.item()) + b) % capacity)

		for probe in range(capacity):
			slot = (current_bucket + probe) % capacity
			current_eff = float(current_value.abs().item())
			slot_empty = bool((out_gamma[slot] <= 0).item())
			slot_eff = float("-inf") if slot_empty else float(out_values[slot].abs().item())
			current_wins = slot_empty or current_eff > slot_eff or (current_eff == slot_eff and current_dib > int(out_dib[slot].item()))

			if not current_wins:
				continue

			displaced_value = out_values[slot].clone()
			displaced_gamma = out_gamma[slot].clone()
			displaced_dib = int(out_dib[slot].item())

			out_values[slot] = current_value
			out_gamma[slot] = current_gamma
			out_dib[slot] = current_dib
			last_inserted_value = current_value
			last_inserted_gamma = current_gamma

			if slot_empty:
				break

			current_value = displaced_value
			current_gamma = displaced_gamma
			current_dib = displaced_dib + 1

	if last_inserted_value is not None:
		last_inserted_value_abs = last_inserted_value.abs()
		if last_inserted_gamma is not None and last_inserted_value_abs < cutoff_bound_slow_mag:
			cutoff_bound_slow_mag = last_inserted_value_abs
			cutoff_bound_slow_gamma = last_inserted_gamma

	return out_values, out_dib, out_gamma, cutoff_bound_slow_mag, cutoff_bound_slow_gamma


def python_rh_write_batched(
	table_values: torch.Tensor,
	table_dib: torch.Tensor,
	table_gamma: torch.Tensor,
	incoming_values: torch.Tensor,
	incoming_indices: torch.Tensor,
	incoming_gammas: torch.Tensor,
	cutoff_bound_slow_mag: torch.Tensor,
	cutoff_bound_slow_gamma: torch.Tensor,
	capacity: int,
	a: int = 1,
	b: int = 0,
):
	if table_values.dim() != 2:
		raise ValueError("table_values must be shaped (B, C)")
	if incoming_values.dim() != 2:
		raise ValueError("incoming_values must be shaped (B, K)")

	out_values = table_values.clone()
	out_dib = table_dib.clone()
	out_gamma = table_gamma.clone()
	out_cutoff_bound_slow_mag = cutoff_bound_slow_mag.clone()
	out_cutoff_bound_slow_gamma = cutoff_bound_slow_gamma.clone()

	for batch_index in range(table_values.size(0)):
		batch_out = _python_rh_write_single(
			out_values[batch_index],
			out_dib[batch_index],
			out_gamma[batch_index],
			incoming_values[batch_index],
			incoming_indices[batch_index],
			incoming_gammas[batch_index],
			out_cutoff_bound_slow_mag[batch_index],
			out_cutoff_bound_slow_gamma[batch_index],
			capacity,
			a,
			b,
		)
		out_values[batch_index], out_dib[batch_index], out_gamma[batch_index], out_cutoff_bound_slow_mag[batch_index], out_cutoff_bound_slow_gamma[batch_index] = batch_out

	return out_values, out_dib, out_gamma, out_cutoff_bound_slow_mag, out_cutoff_bound_slow_gamma