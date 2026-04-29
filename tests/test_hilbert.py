from __future__ import annotations

import torch

from rh_memory.hilbert import (
    hilbert_flatten_images,
    hilbert_permutation,
    hilbert_unflatten_images,
    inverse_hilbert_permutation,
)


def test_hilbert_permutation_covers_square_indices() -> None:
    perm = hilbert_permutation(32)
    inverse = inverse_hilbert_permutation(32)

    assert perm.shape == (1024,)
    assert torch.equal(torch.sort(perm).values, torch.arange(1024))
    assert torch.equal(perm[inverse], torch.arange(1024))


def test_hilbert_flatten_round_trips_images() -> None:
    images = torch.arange(2 * 32 * 32, dtype=torch.float32).reshape(2, 1, 32, 32)

    sequence = hilbert_flatten_images(images)
    restored = hilbert_unflatten_images(sequence)

    assert sequence.shape == (2, 1, 1024)
    assert torch.equal(restored, images)


def test_hilbert_permutation_is_deterministic() -> None:
    assert torch.equal(hilbert_permutation(32), hilbert_permutation(32))
