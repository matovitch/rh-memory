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
	capacity: int,
	a: int = 1,
):
	"""Apply one RH write batch to a single table.

	The table starts zero-filled and is always treated as filled after
	initialization. Writes rank items by raw absolute magnitude; gamma is stored so
	`advance_time` can decay the table and cutoff bound over elapsed time.
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

	for value, index, gamma in zip(incoming_values, incoming_indices, incoming_gammas):
		if float(value.abs().item()) == 0.0:
			break
			
		current_value = value
		current_gamma = gamma
		current_dib = 0
		current_index = int(index.item())
		current_bucket = int((a * (current_index // capacity) + current_index) % capacity)

		for probe in range(capacity):
			slot = (current_bucket + probe) % capacity
			current_eff = float(current_value.abs().item())
			slot_eff = float(out_values[slot].abs().item())
			current_wins = current_eff > slot_eff or (current_eff == slot_eff and current_dib > int(out_dib[slot].item()))

			if not current_wins:
				continue

			displaced_value = out_values[slot].clone()
			displaced_gamma = out_gamma[slot].clone()
			displaced_dib = int(out_dib[slot].item())

			out_values[slot] = current_value
			out_gamma[slot] = current_gamma
			out_dib[slot] = current_dib

			current_value = displaced_value
			current_gamma = displaced_gamma
			current_dib = displaced_dib + 1

	return out_values, out_dib, out_gamma


def python_rh_write_batched(
	table_values: torch.Tensor,
	table_dib: torch.Tensor,
	table_gamma: torch.Tensor,
	incoming_values: torch.Tensor,
	incoming_indices: torch.Tensor,
	incoming_gammas: torch.Tensor,
	capacity: int,
	a: int = 1,
):
	if table_values.dim() != 2:
		raise ValueError("table_values must be shaped (B, C)")
	if incoming_values.dim() != 2:
		raise ValueError("incoming_values must be shaped (B, K)")

	out_values = table_values.clone()
	out_dib = table_dib.clone()
	out_gamma = table_gamma.clone()

	for batch_index in range(table_values.size(0)):
		batch_out = _python_rh_write_single(
			out_values[batch_index],
			out_dib[batch_index],
			out_gamma[batch_index],
			incoming_values[batch_index],
			incoming_indices[batch_index],
			incoming_gammas[batch_index],
			capacity,
			a,
		)
		out_values[batch_index], out_dib[batch_index], out_gamma[batch_index] = batch_out

	return out_values, out_dib, out_gamma


def python_fast_rh_write_batched(
	table_values: torch.Tensor,
	table_dib: torch.Tensor,
	table_gamma: torch.Tensor,
	incoming_values: torch.Tensor,
	incoming_gammas: torch.Tensor,
	a: int,
	k: int,
):
	if table_values.dim() != 2:
		raise ValueError("table_values must be shaped (B, C)")
	if incoming_values.dim() != 2:
		raise ValueError("incoming_values must be shaped (B, n)")

	batch_size, n = incoming_values.shape
	C = table_values.size(1)
	if n % C != 0:
		raise ValueError("n must be divisible by C")
	stride = n // C

	inc_vals = incoming_values.view(batch_size, stride, C).clone()
	inc_gams = incoming_gammas.view(batch_size, stride, C).clone()

	# Initial alignment
	for r in range(stride):
		shift = (r * a) % C
		if shift != 0:
			inc_vals[:, r, :] = torch.roll(inc_vals[:, r, :], shifts=shift, dims=-1)
			inc_gams[:, r, :] = torch.roll(inc_gams[:, r, :], shifts=shift, dims=-1)

	out_values = table_values.clone()
	out_dib = table_dib.clone()
	out_gamma = table_gamma.clone()

	for step in range(k):
		abs_vals = inc_vals.abs()
		current_winners_abs, winner_indices = torch.max(abs_vals, dim=1)

		incumbent_abs = out_values.abs()
		mask = current_winners_abs > incumbent_abs

		extracted_vals = torch.gather(inc_vals, 1, winner_indices.unsqueeze(1)).squeeze(1)
		extracted_gams = torch.gather(inc_gams, 1, winner_indices.unsqueeze(1)).squeeze(1)

		out_values = torch.where(mask, extracted_vals, out_values)
		out_gamma = torch.where(mask, extracted_gams, out_gamma)
		out_dib = torch.where(mask, torch.full_like(out_dib, step, dtype=out_dib.dtype, device=out_dib.device), out_dib)

		stride_indices = torch.arange(stride, device=inc_vals.device).view(1, stride, 1)
		is_winning_and_updating = (stride_indices == winner_indices.unsqueeze(1)) & mask.unsqueeze(1)

		inc_vals.masked_fill_(is_winning_and_updating, 0.0)
		inc_gams.masked_fill_(is_winning_and_updating, 0.0)

		if step < k - 1:
			inc_vals = torch.roll(inc_vals, shifts=1, dims=-1)
			inc_gams = torch.roll(inc_gams, shifts=1, dims=-1)

	return out_values, out_dib, out_gamma