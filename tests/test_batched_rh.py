import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_batched_python_and_cpp_write_match():
    from rh_memory import (
        BatchedSlowMemoryState,
        compute_write_gammas,
    )

    torch.manual_seed(0)

    batch_size = 3
    capacity = 8
    python_state = BatchedSlowMemoryState.empty(batch_size, capacity)
    cpp_state = BatchedSlowMemoryState.empty(batch_size, capacity)

    incoming_values = torch.tensor(
        [
            [5.0, 4.0, 3.0, 2.0, 1.0, 0.9, 0.8, 0.7],
            [1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8],
            [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0],
        ]
    )
    incoming_indices = torch.tensor(
        [
            [7, 6, 5, 4, 3, 2, 1, 0],
            [1, 3, 5, 7, 0, 2, 4, 6],
            [2, 4, 6, 0, 1, 3, 5, 7],
        ],
        dtype=torch.long,
    )
    incoming_gammas = compute_write_gammas(incoming_values, alpha=0.2, epsilon=0.05)

    python_state.advance_time(4)
    cpp_state.advance_time(4)

    python_state.write_python(
        incoming_values,
        incoming_indices,
        incoming_gammas,
        capacity,
        1,
        0,
    )
    cpp_state.write_cpp(
        incoming_values,
        incoming_indices,
        incoming_gammas,
        capacity,
        1,
        0,
    )

    assert torch.allclose(python_state.values, cpp_state.values)
    assert torch.allclose(python_state.dib, cpp_state.dib)
    assert torch.allclose(python_state.gamma, cpp_state.gamma)
    assert torch.allclose(python_state.cutoff_bound_slow_mag, cpp_state.cutoff_bound_slow_mag)
    assert torch.allclose(python_state.cutoff_bound_slow_gamma, cpp_state.cutoff_bound_slow_gamma)
    occupied = python_state.gamma > 0
    assert torch.all(python_state.values[occupied] >= 0)


def test_full_table_write_uses_fast_path_and_matches_python():
    from rh_memory import (
        BatchedSlowMemoryState,
        compute_write_gammas,
    )

    batch_size = 2
    capacity = 4
    python_state = BatchedSlowMemoryState.empty(batch_size, capacity)
    cpp_state = BatchedSlowMemoryState.empty(batch_size, capacity)

    python_state.values.copy_(torch.tensor([[4.0, 3.0, 2.0, 1.0], [1.5, 2.5, 3.5, 4.5]]))
    cpp_state.values.copy_(python_state.values)
    python_state.dib.copy_(torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.long))
    cpp_state.dib.copy_(python_state.dib)
    python_state.gamma.fill_(0.9)
    cpp_state.gamma.fill_(0.9)

    incoming_values = torch.tensor([[5.0, 0.5, 4.0], [2.0, 6.0, 1.0]])
    incoming_indices = torch.tensor([[0, 1, 2], [3, 2, 1]], dtype=torch.long)
    incoming_gammas = compute_write_gammas(incoming_values, alpha=0.3, epsilon=0.1)

    python_state.write_python(
        incoming_values,
        incoming_indices,
        incoming_gammas,
        capacity,
        1,
        0,
    )
    cpp_state.write_cpp(
        incoming_values,
        incoming_indices,
        incoming_gammas,
        capacity,
        1,
        0,
    )

    assert torch.allclose(python_state.values, cpp_state.values)
    assert torch.allclose(python_state.dib, cpp_state.dib)
    assert torch.allclose(python_state.gamma, cpp_state.gamma)
    assert torch.allclose(python_state.cutoff_bound_slow_mag, cpp_state.cutoff_bound_slow_mag)
    assert torch.allclose(python_state.cutoff_bound_slow_gamma, cpp_state.cutoff_bound_slow_gamma)


def test_first_write_can_fill_empty_table():
    from rh_memory import (
        BatchedSlowMemoryState,
        compute_write_gammas,
    )

    batch_size = 2
    capacity = 4
    python_state = BatchedSlowMemoryState.empty(batch_size, capacity)
    cpp_state = BatchedSlowMemoryState.empty(batch_size, capacity)

    incoming_values = torch.tensor(
        [
            [8.0, 7.0, 6.0, 5.0],
            [4.0, 3.0, 2.0, 1.0],
        ]
    )
    incoming_indices = torch.tensor(
        [
            [0, 1, 2, 3],
            [3, 2, 1, 0],
        ],
        dtype=torch.long,
    )
    incoming_gammas = compute_write_gammas(incoming_values, alpha=0.3, epsilon=0.1)

    python_state.write_python(
        incoming_values,
        incoming_indices,
        incoming_gammas,
        capacity,
        1,
        0,
    )
    cpp_state.write_cpp(
        incoming_values,
        incoming_indices,
        incoming_gammas,
        capacity,
        1,
        0,
    )

    assert torch.allclose(python_state.values, cpp_state.values)
    assert torch.allclose(python_state.dib, cpp_state.dib)
    assert torch.allclose(python_state.gamma, cpp_state.gamma)
    assert torch.all(python_state.gamma > 0)


def test_advance_time_decays_batched_state():
    from rh_memory import BatchedSlowMemoryState

    state = BatchedSlowMemoryState.empty(batch_size=2, capacity=3)
    state.values.fill_(2.0)
    state.gamma.fill_(0.9)
    state.cutoff_bound_slow_mag.fill_(2.0)
    state.cutoff_bound_slow_gamma.fill_(0.9)

    state.advance_time(3)

    expected_value = 2.0 * (0.9 ** 3)
    assert torch.allclose(state.values, torch.full_like(state.values, expected_value))
    assert torch.allclose(state.cutoff_bound_slow_mag, torch.full_like(state.cutoff_bound_slow_mag, expected_value))
    assert torch.allclose(state.cutoff_bound_slow_gamma, torch.full_like(state.cutoff_bound_slow_gamma, 0.9))


def test_advance_time_recomputes_exact_cutoff_for_nonuniform_gammas():
    from rh_memory import BatchedSlowMemoryState

    state = BatchedSlowMemoryState.empty(batch_size=1, capacity=3)
    state.values.copy_(torch.tensor([[4.0, 3.0, 2.0]]))
    state.gamma.copy_(torch.tensor([[0.5, 0.9, 0.8]]))
    state.cutoff_bound_slow_mag.copy_(torch.tensor([3.0]))
    state.cutoff_bound_slow_gamma.copy_(torch.tensor([0.9]))

    state.advance_time(1)

    expected_values = torch.tensor([[2.0, 2.7, 1.6]])
    assert torch.allclose(state.values, expected_values)
    assert torch.allclose(state.cutoff_bound_slow_mag, torch.tensor([1.6]))
    assert torch.allclose(state.cutoff_bound_slow_gamma, torch.tensor([0.8]))