from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PipelineConfig:
    """Shared configuration for synthetic RH-memory experiment pipelines."""

    n: int
    C: int
    batch_size: int
    seed: int
    fast_k: float
    harmonic_decay: float = 0.65
    harmonic_amp_threshold: float = 0.1
    max_harmonics: int = 64

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError(f"N must be positive, got {self.n}")
        if self.C <= 0:
            raise ValueError(f"C must be positive, got {self.C}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.n % self.C != 0:
            raise ValueError(f"Grouped permutation mode requires N % C == 0, got N={self.n}, C={self.C}")
        if self.fast_k <= 0:
            raise ValueError(f"fast_k must be positive, got {self.fast_k}")
        if self.max_harmonics <= 0:
            raise ValueError(f"max_harmonics must be positive, got {self.max_harmonics}")

    @property
    def sequence_length(self) -> int:
        return self.n

    @property
    def bucket_count(self) -> int:
        return self.C

    @property
    def stride(self) -> int:
        return self.n // self.C

    @property
    def k_eff(self) -> int:
        return max(1, int(self.fast_k * math.log(self.C)))

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, int | float]) -> "PipelineConfig":
        return cls(
            n=int(data["n"]),
            C=int(data["C"]),
            batch_size=int(data["batch_size"]),
            seed=int(data["seed"]),
            fast_k=float(data["fast_k"]),
            harmonic_decay=float(data.get("harmonic_decay", 0.65)),
            harmonic_amp_threshold=float(data.get("harmonic_amp_threshold", 0.1)),
            max_harmonics=int(data.get("max_harmonics", 64)),
        )
