from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_import_package():
    import rh_memory

    assert rh_memory.__doc__ is not None
