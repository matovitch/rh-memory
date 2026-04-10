from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_import_package():
    import rh_memory

    assert rh_memory.__doc__ is not None


def test_cpu_ops_module_import():
    from rh_memory import extension_available

    assert isinstance(extension_available(), bool)


def test_gamma_helper_shape():
    import torch

    from rh_memory import compute_write_gammas

    values = torch.tensor([1.0, 2.0, 3.0])
    gammas = compute_write_gammas(values, alpha=0.5, epsilon=0.1)

    assert gammas.shape == values.shape
    assert torch.all(gammas >= 0.1)
    assert torch.all(gammas < 1.0)
