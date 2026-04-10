"""Fast and slow memory state utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ._cpu_ops import cpu_rh_advance_time, cpu_rh_write, cpu_rh_write_batched
from ._python_ops import python_rh_write_batched, python_fast_rh_write_batched


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
	
	if isinstance(cutoff_bound_slow_mag, float):
		threshold = torch.full((sorted_values.size(0), 1), cutoff_bound_slow_mag, device=sorted_values.device, dtype=sorted_values.dtype)
	else:
		threshold = cutoff_bound_slow_mag.unsqueeze(1).to(sorted_values.device, sorted_values.dtype) # type: ignore

	active_mask = effective_magnitudes >= threshold
	lengths = active_mask.sum(dim=1)
	
	if lengths.numel() == 0 or lengths.max().item() == 0:
		return (
			torch.empty((sorted_values.size(0), 0), device=sorted_values.device, dtype=sorted_values.dtype),
			torch.empty((sorted_values.size(0), 0), device=sorted_values.device, dtype=torch.long),
			torch.empty((sorted_values.size(0), 0), device=sorted_values.device, dtype=sorted_values.dtype),
		)
		
	max_cutoff = int(lengths.max().item())
	
	padded_values  = sorted_values    [:, :max_cutoff] * active_mask[:, :max_cutoff]
	padded_indices = sort_permutation [:, :max_cutoff] * active_mask[:, :max_cutoff]
	padded_gammas  = sorted_gammas    [:, :max_cutoff] * active_mask[:, :max_cutoff]
	
	return padded_values, padded_indices, padded_gammas

@dataclass
class BatchedFastMemoryState:
	values: torch.Tensor
	dib: torch.Tensor
	gamma: torch.Tensor
	alpha: float = 1.0
	epsilon: float = 0.05
	a: int = 1
	k: int = 1

	@classmethod
	def empty(
		cls,
		batch_size: int,
		capacity: int,
		dtype: torch.dtype = torch.float32,
		device: torch.device | str = "cpu",
		alpha: float = 1.0,
		epsilon: float = 0.05,
		a: int = 1,
		k: int = 1,
	) -> "BatchedFastMemoryState":
		values = torch.zeros(batch_size, capacity, dtype=dtype, device=device)
		dib = torch.zeros(batch_size, capacity, dtype=torch.long, device=device)
		gamma = torch.zeros(batch_size, capacity, dtype=dtype, device=device)
		return cls(values=values, dib=dib, gamma=gamma, alpha=alpha, epsilon=epsilon, a=a, k=k)

	def write(
		self,
		incoming_values: torch.Tensor,
	) -> "BatchedFastMemoryState":
		self.advance_time()
		return self.write_python(
			incoming_values,
		)

	def write_python(
		self,
		incoming_values: torch.Tensor,
	) -> "BatchedFastMemoryState":
		incoming_gammas = compute_write_gammas(incoming_values, self.alpha, self.epsilon)
		self.values, self.dib, self.gamma = python_fast_rh_write_batched(
			self.values,
			self.dib,
			self.gamma,
			incoming_values,
			incoming_gammas,
			self.a,
			self.k,
		)
		return self

	def advance_time(self) -> "BatchedFastMemoryState":
		self.values *= self.gamma
		return self

@dataclass
class BatchedSlowMemoryState:
	values: torch.Tensor
	dib: torch.Tensor
	gamma: torch.Tensor
	cutoff_bound_slow_mag: torch.Tensor
	cutoff_bound_slow_gamma: torch.Tensor
	alpha: float = 1.0
	epsilon: float = 0.05
	a: int = 1
	use_cpp: bool = True

	@property
	def capacity(self) -> int:
		return self.values.size(1)

	@classmethod
	def empty(
		cls,
		batch_size: int,
		capacity: int,
		dtype: torch.dtype = torch.float32,
		device: torch.device | str = "cpu",
		alpha: float = 1.0,
		epsilon: float = 0.05,
		a: int = 1,
		use_cpp: bool = True,
	) -> "BatchedSlowMemoryState":
		values = torch.zeros(batch_size, capacity, dtype=dtype, device=device)
		dib = torch.zeros(batch_size, capacity, dtype=torch.long, device=device)
		gamma = torch.zeros(batch_size, capacity, dtype=dtype, device=device)
		cutoff_bound_slow_mag = torch.zeros(batch_size, dtype=dtype, device=device)
		cutoff_bound_slow_gamma = torch.zeros(batch_size, dtype=dtype, device=device)
		return cls(values=values, dib=dib, gamma=gamma, cutoff_bound_slow_mag=cutoff_bound_slow_mag, cutoff_bound_slow_gamma=cutoff_bound_slow_gamma, alpha=alpha, epsilon=epsilon, a=a, use_cpp=use_cpp)

	def write(
		self,
		incoming_values: torch.Tensor,
		delta_steps: int,
	) -> "BatchedSlowMemoryState":
		
		self.advance_time(delta_steps)
		
		incoming_gammas = compute_write_gammas(incoming_values, self.alpha, self.epsilon)

		abs_vals = incoming_values.abs()
		_, sort_permutation = torch.sort(abs_vals, dim=-1, descending=True)
		sorted_values = torch.gather(incoming_values, dim=1, index=sort_permutation)
		sorted_gammas = torch.gather(incoming_gammas, dim=1, index=sort_permutation)

		trunc_vals, trunc_idx, trunc_gammas = truncate_sorted_write(
			sorted_values,
			sort_permutation,
			sorted_gammas,
			self.cutoff_bound_slow_mag,
		)

		if trunc_vals.size(1) > 0:
			if self.use_cpp:
				self.write_cpp(trunc_vals, trunc_idx, trunc_gammas)
			else:
				self.write_python(trunc_vals, trunc_idx, trunc_gammas)

		return self

	def write_python(
		self,
		incoming_values: torch.Tensor,
		incoming_indices: torch.Tensor,
		incoming_gammas: torch.Tensor,
	) -> "BatchedSlowMemoryState":
		self.values, self.dib, self.gamma = rh_write_batched_python(
			self.values,
			self.dib,
			self.gamma,
			incoming_values,
			incoming_indices,
			incoming_gammas,
			capacity=self.capacity,
			a=self.a,
		)
		return self

	def write_cpp(
		self,
		incoming_values: torch.Tensor,
		incoming_indices: torch.Tensor,
		incoming_gammas: torch.Tensor,
	) -> "BatchedSlowMemoryState":
		self.values, self.dib, self.gamma = rh_write_batched_cpp(
			self.values,
			self.dib,
			self.gamma,
			incoming_values,
			incoming_indices,
			incoming_gammas,
			capacity=self.capacity,
			a=self.a,
		)
		return self

	def advance_time(
		self,
		delta_steps: int,
	) -> "BatchedSlowMemoryState":
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
		capacity: int,
		a: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	return python_rh_write_batched(
		self_values,
		self_dib,
		self_gamma,
		incoming_values,
		incoming_indices,
		incoming_gammas,
		capacity,
		a,
	)


def rh_write_batched_cpp(
		self_values: torch.Tensor,
		self_dib: torch.Tensor,
		self_gamma: torch.Tensor,
		incoming_values: torch.Tensor,
		incoming_indices: torch.Tensor,
		incoming_gammas: torch.Tensor,
		capacity: int,
		a: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	return cpu_rh_write_batched(
		self_values,
		self_dib,
		self_gamma,
		incoming_values,
		incoming_indices,
		incoming_gammas,
		capacity,
		a,
	)
