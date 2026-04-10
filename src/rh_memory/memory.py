"""Fast and slow memory state utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ._cpu_ops import cpu_rh_advance_time, cpu_rh_write, cpu_rh_write_batched
from ._python_ops import python_rh_write_batched


def compute_write_gammas(
    values: torch.Tensor,
    alpha: float,
	epsilon: float = 0.05,
) -> torch.Tensor:
	if not 0.0 <= epsilon < 1.0:
		raise ValueError("epsilon must be in [0, 1)")

	return epsilon + (1.0 - epsilon) * (1.0 - torch.exp(-alpha * values.abs()))


def truncate_sorted_write(
	sorted_values: torch.Tensor,
	sort_permutation: torch.Tensor,
	sorted_gammas: torch.Tensor,
	cutoff_bound_slow_mag: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	effective_magnitudes = sorted_values.abs()
	threshold = torch.as_tensor(
		cutoff_bound_slow_mag,
		device=effective_magnitudes.device,
		dtype=effective_magnitudes.dtype,
	)
	cutoff = torch.searchsorted(-effective_magnitudes, -threshold).item()
	return (
		sorted_values[:cutoff],
		sort_permutation[:cutoff],
		sorted_gammas[:cutoff],
	)

@dataclass
class BatchedSlowMemoryState:
	values: torch.Tensor
	dib: torch.Tensor
	gamma: torch.Tensor
	cutoff_bound_slow_mag: torch.Tensor
	cutoff_bound_slow_gamma: torch.Tensor

	@classmethod
	def empty(
		cls,
		batch_size: int,
		capacity: int,
		dtype: torch.dtype = torch.float32,
		device: torch.device | str = "cpu",
	) -> "BatchedSlowMemoryState":
		values = torch.zeros(batch_size, capacity, dtype=dtype, device=device)
		dib = torch.zeros(batch_size, capacity, dtype=torch.long, device=device)
		gamma = torch.zeros(batch_size, capacity, dtype=dtype, device=device)
		cutoff_bound_slow_mag = torch.zeros(batch_size, dtype=dtype, device=device)
		cutoff_bound_slow_gamma = torch.zeros(batch_size, dtype=dtype, device=device)
		return cls(values=values, dib=dib, gamma=gamma, cutoff_bound_slow_mag=cutoff_bound_slow_mag, cutoff_bound_slow_gamma=cutoff_bound_slow_gamma)

	def write_python(
		self,
		incoming_values: torch.Tensor,
		incoming_indices: torch.Tensor,
		incoming_gammas: torch.Tensor,
		capacity: int,
		a: int = 1,
		b: int = 0,
	) -> "BatchedSlowMemoryState":
		self.values, self.dib, self.gamma, self.cutoff_bound_slow_mag, self.cutoff_bound_slow_gamma = rh_write_batched_python(
			self.values,
			self.dib,
			self.gamma,
			incoming_values,
			incoming_indices,
			incoming_gammas,
			self.cutoff_bound_slow_mag,
			self.cutoff_bound_slow_gamma,
			capacity,
			a,
			b,
		)
		return self

	def write_cpp(
		self,
		incoming_values: torch.Tensor,
		incoming_indices: torch.Tensor,
		incoming_gammas: torch.Tensor,
		capacity: int,
		a: int = 1,
		b: int = 0,
	) -> "BatchedSlowMemoryState":
		self.values, self.dib, self.gamma, self.cutoff_bound_slow_mag, self.cutoff_bound_slow_gamma = rh_write_batched_cpp(
			self.values,
			self.dib,
			self.gamma,
			incoming_values,
			incoming_indices,
			incoming_gammas,
			self.cutoff_bound_slow_mag,
			self.cutoff_bound_slow_gamma,
			capacity,
			a,
			b,
		)
		return self

	def advance_time(
		self,
		delta_steps: int,
	) -> "BatchedSlowMemoryState":
		if delta_steps > 0:
			self.values, self.gamma, self.cutoff_bound_slow_mag, self.cutoff_bound_slow_gamma = cpu_rh_advance_time(
				self.values,
				self.gamma,
				self.cutoff_bound_slow_mag,
				self.cutoff_bound_slow_gamma,
				delta_steps,
			)
		return self

def rh_write_batched_python(
		self_values: torch.Tensor,
		self_dib: torch.Tensor,
		self_gamma: torch.Tensor,
		incoming_values: torch.Tensor,
		incoming_indices: torch.Tensor,
		incoming_gammas: torch.Tensor,
		cutoff_bound_slow_mag: torch.Tensor,
		cutoff_bound_slow_gamma: torch.Tensor,
		capacity: int,
		a: int = 1,
		b: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	return python_rh_write_batched(
		self_values,
		self_dib,
		self_gamma,
		incoming_values,
		incoming_indices,
		incoming_gammas,
		cutoff_bound_slow_mag,
		cutoff_bound_slow_gamma,
		capacity,
		a,
		b,
	)


def rh_write_batched_cpp(
		self_values: torch.Tensor,
		self_dib: torch.Tensor,
		self_gamma: torch.Tensor,
		incoming_values: torch.Tensor,
		incoming_indices: torch.Tensor,
		incoming_gammas: torch.Tensor,
		cutoff_bound_slow_mag: torch.Tensor,
		cutoff_bound_slow_gamma: torch.Tensor,
		capacity: int,
		a: int = 1,
		b: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	return cpu_rh_write_batched(
		self_values,
		self_dib,
		self_gamma,
		incoming_values,
		incoming_indices,
		incoming_gammas,
		cutoff_bound_slow_mag,
		cutoff_bound_slow_gamma,
		capacity,
		a,
		b,
	)
