"""Evaluate energy-to-image directional flow checkpoint distributions."""

from __future__ import annotations

from pathlib import Path

from directional_flow_distribution_eval import main


if __name__ == "__main__":
    main(
        default_checkpoint=Path("scripts/checkpoints/reflow_energy_to_image_checkpoint.pt"),
        fixed_direction="energy-to-image",
    )