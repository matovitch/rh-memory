"""Evaluate image-to-energy directional flow checkpoint distributions."""

from __future__ import annotations

from pathlib import Path

from directional_flow_distribution_eval import main


if __name__ == "__main__":
    main(
        default_checkpoint=Path("scripts/checkpoints/reflow_image_to_energy_checkpoint.pt"),
        fixed_direction="image-to-energy",
    )
