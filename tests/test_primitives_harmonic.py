from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rh_memory.pipeline.primitives_harmonic import harmonic_raw_batch


def test_harmonic_raw_batch_shape_dtype_and_finite():
    x = harmonic_raw_batch(
        chunk_size=4,
        n=32,
        device="cpu",
        harmonic_decay=0.65,
        harmonic_amp_threshold=0.1,
        max_harmonics=8,
    )
    assert x.shape == (4, 32)
    assert x.dtype == torch.float32
    assert torch.isfinite(x).all()


def test_harmonic_raw_batch_reproducible_under_seed():
    torch.manual_seed(123)
    a = harmonic_raw_batch(
        chunk_size=2,
        n=16,
        device="cpu",
        harmonic_decay=0.65,
        harmonic_amp_threshold=0.1,
        max_harmonics=8,
    )
    torch.manual_seed(123)
    b = harmonic_raw_batch(
        chunk_size=2,
        n=16,
        device="cpu",
        harmonic_decay=0.65,
        harmonic_amp_threshold=0.1,
        max_harmonics=8,
    )
    assert torch.equal(a, b)
