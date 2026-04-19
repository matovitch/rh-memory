"""Pytest hooks.

``TRITON_INTERPRET`` must be set **before** ``rh_memory`` (and thus ``@triton.jit``)
import time; otherwise Triton builds ``JITFunction`` and still requires an active GPU
driver even for CPU-only execution.
"""

from __future__ import annotations


def pytest_configure(config):
    import os

    import torch

    if not torch.cuda.is_available():
        os.environ["TRITON_INTERPRET"] = "1"
