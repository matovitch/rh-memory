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
    )
    cpp_state.write_cpp(
        incoming_values,
        incoming_indices,
        incoming_gammas,
        capacity,
        1,
    )

    assert torch.allclose(python_state.values, cpp_state.values)
    assert torch.allclose(python_state.dib, cpp_state.dib)
    assert torch.allclose(python_state.gamma, cpp_state.gamma)
    assert torch.allclose(python_state.cutoff_bound_slow_mag, cpp_state.cutoff_bound_slow_mag)
    assert torch.allclose(python_state.cutoff_bound_slow_gamma, cpp_state.cutoff_bound_slow_gamma)
    occupied = python_state.gamma > 0
    assert torch.all(python_state.values[occupied] >= 0)


def test_write_on_prepopulated_table_matches_python():
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
    )
    cpp_state.write_cpp(
        incoming_values,
        incoming_indices,
        incoming_gammas,
        capacity,
        1,
    )

    assert torch.allclose(python_state.values, cpp_state.values)
    assert torch.allclose(python_state.dib, cpp_state.dib)
    assert torch.allclose(python_state.gamma, cpp_state.gamma)
    assert torch.allclose(python_state.cutoff_bound_slow_mag, cpp_state.cutoff_bound_slow_mag)
    assert torch.allclose(python_state.cutoff_bound_slow_gamma, cpp_state.cutoff_bound_slow_gamma)


def test_zero_initialized_table_writes_match_python_and_cpp():
    from rh_memory import (
        BatchedSlowMemoryState,
        compute_write_gammas,
    )

    batch_size = 2
    capacity = 4
    python_state = BatchedSlowMemoryState.empty(batch_size, capacity)
    cpp_state = BatchedSlowMemoryState.empty(batch_size, capacity)

    assert torch.all(python_state.values == 0)
    assert torch.all(python_state.dib == 0)
    assert torch.all(python_state.gamma == 0)

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
    )
    cpp_state.write_cpp(
        incoming_values,
        incoming_indices,
        incoming_gammas,
        capacity,
        1,
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


def test_fast_memory_basic_routing_and_collision():
    from rh_memory import BatchedFastMemoryState
    
    batch_size = 1
    capacity = 4
    n = 8
    state = BatchedFastMemoryState.empty(batch_size, capacity)
    
    # We set a = 1
    # Elements:
    # index 0, 1, 2, 3 in stride 0 -> bucket 0, 1, 2, 3 (since r=0, shift=0)
    # index 4, 5, 6, 7 in stride 1 -> bucket 1, 2, 3, 0 (r=1, a=1, shift=1)
    
    # Let's put a highly valued element at n=0 -> bucket 0
    # And one at n=7 -> bucket 0 too
    # If the one at n=7 is smaller, it bounces to bucket 1
    # If the one at n=7 is bigger, it takes bucket 0, and n=0 would bounce? 
    # Wait, the loop processes all contenders for bucket c simultaneously.
    
    incoming_values = torch.tensor([[10.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
    incoming_gammas = torch.tensor([[0.9] * 8])
    
    # step 0:
    # candidates for bucket 0: n=0 (value 10.0) from stride 0, and n=7 (value 8.0) from stride 1
    # Max is 10.0. Bucket 0 gets 10.0. Element n=0 gets zeroed out.
    # candidates for bucket 1: n=1 (value 2.0) from stride 0, and n=4 (value 5.0) from stride 1
    # Max is 5.0. Bucket 1 gets 5.0. Element n=4 gets zeroed out.
    # candidates for bucket 2: n=2 (value 3.0) from stride 0, and n=5 (value 6.0) from stride 1
    # Max is 6.0. Bucket 2 gets 6.0. Element n=5 gets zeroed out.
    # candidates for bucket 3: n=3 (value 4.0) from stride 0, and n=6 (value 7.0) from stride 1
    # Max is 7.0. Bucket 3 gets 7.0. Element n=6 gets zeroed out.
    
    # This leaves n=1 (2.0), n=2 (3.0), n=3 (4.0), n=7 (8.0).
    # Roll by 1.
    # Now: candidate for bucket 1 is 8.0 (from n=7's row) and 0 (from n=0's row) -> bucket 1 incumbent is 5.0. 8.0 > 5.0, so 8.0 takes bucket 1.
    # candidate for bucket 2 is 2.0 (from n=1) -> bucket 2 incumbent is 6.0. No update.
    # candidate for bucket 3 is 3.0 (from n=2) -> bucket 3 incumbent is 7.0. No update.
    # candidate for bucket 0 is 4.0 (from n=3) -> bucket 0 incumbent is 10.0. No update.
    
    # Result should be roughly: [10.0, 8.0, 6.0, 7.0]
    
    state.write_python(incoming_values, incoming_gammas, a=1, k=2)
    expected_values = torch.tensor([[10.0, 8.0, 6.0, 7.0]])
    expected_dib = torch.tensor([[0, 1, 0, 0]], dtype=torch.long)
    
    assert torch.allclose(state.values, expected_values)
    assert torch.allclose(state.dib, expected_dib)


def test_fast_memory_erase_logic():
    from rh_memory import BatchedFastMemoryState
    
    state = BatchedFastMemoryState.empty(1, 2)
    # n = 4 (stride = 2). capacity = 2. a = 1
    # n=0, n=1 (stride 0)
    # n=2, n=3 (stride 1)
    
    # We want to test that a winner is zeroed and does NOT overwrite next buckets.
    # n=0: 10.0, n=1: 1.0, n=2: 5.0, n=3: 5.0
    # step 0:
    # bucket 0 candidates: 10.0 (n=0) and 5.0 (n=3) -> 10.0 wins (bucket 0).
    # bucket 1 candidates: 1.0 (n=1) and 5.0 (n=2) -> 5.0 wins (bucket 1).
    # Winners zeroed: n=0 and n=2.
    
    # step 1: (rolled by 1)
    # bucket 1 candidates: 0.0 (old n=0) and 5.0 (n=3). incumbent bucket 1 is 5.0. 5.0 > 5.0 is False, no overwrite.
    # bucket 0 candidates: 1.0 (old n=1) and 0.0 (old n=2). incumbent bucket 0 is 10.0. no overwrite.
    
    # If the erase didn't work, old n=0 (10.0) would roll to bucket 1 and overwrite the 5.0 there!
    incoming_values = torch.tensor([[10.0, 1.0, 5.0, 5.0]])
    incoming_gammas = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    
    state.write_python(incoming_values, incoming_gammas, a=1, k=3)
    
    expected_values = torch.tensor([[10.0, 5.0]])
    assert torch.allclose(state.values, expected_values)
