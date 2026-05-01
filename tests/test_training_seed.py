import torch

from rh_memory.training_seed import apply_training_seed, resolve_training_seed


def test_argument_seed_takes_precedence():
    training_seed = resolve_training_seed(123, {"seed": 456})

    assert training_seed.seed == 123
    assert training_seed.source == "argument"


def test_checkpoint_seed_is_used_when_argument_missing():
    training_seed = resolve_training_seed(None, {"seed": 456})

    assert training_seed.seed == 456
    assert training_seed.source == "checkpoint"


def test_checkpoint_config_seed_is_used_for_legacy_checkpoints():
    training_seed = resolve_training_seed(None, {"config": {"seed": 789}})

    assert training_seed.seed == 789
    assert training_seed.source == "checkpoint config"


def test_system_random_seed_is_used_when_no_seed_exists():
    training_seed = resolve_training_seed(None, None)

    assert 0 <= training_seed.seed < 2**63
    assert training_seed.source == "system randomness"


def test_apply_training_seed_sets_torch_seed():
    training_seed = apply_training_seed(321)

    assert training_seed.seed == 321
    assert torch.initial_seed() == 321
