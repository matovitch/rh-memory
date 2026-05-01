"""Convert a tar of 32x32 PNG images into grayscale torch tensor shards."""

from __future__ import annotations

import argparse
from pathlib import Path

from rh_memory.image_shards import convert_png_tar_to_grayscale_shards


def parse_image_size(value: str) -> tuple[int, int]:
    parts = value.lower().split("x", maxsplit=1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("image size must use WIDTHxHEIGHT, e.g. 32x32")
    try:
        width, height = int(parts[0]), int(parts[1])
    except ValueError as error:
        raise argparse.ArgumentTypeError("image size dimensions must be integers") from error
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("image size dimensions must be positive")
    return width, height


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tar", type=Path, default=Path("data/images_32x32.tar"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/grayscale_32x32_torch"))
    parser.add_argument("--shard-size", type=int, default=65_536)
    parser.add_argument("--image-size", type=parse_image_size, default=(32, 32))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument("--progress-every", type=int, default=50_000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stats = convert_png_tar_to_grayscale_shards(
        args.tar,
        args.output_dir,
        shard_size=args.shard_size,
        image_size=args.image_size,
        limit=args.limit,
        overwrite=args.overwrite,
        resize=args.resize,
        skip_errors=args.skip_errors,
        progress_every=args.progress_every,
    )
    print(f"wrote {stats.image_count} images across {stats.shard_count} shards to {stats.manifest_path.parent}")
    if stats.skipped_count:
        print(f"skipped {stats.skipped_count} images")
    print(f"manifest: {stats.manifest_path}")


if __name__ == "__main__":
    main()
