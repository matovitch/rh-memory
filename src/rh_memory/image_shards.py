"""Utilities for grayscale image tensor shards."""

from __future__ import annotations

import bisect
import json
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


SHARD_FORMAT = "rh-memory-grayscale-image-shards"
SHARD_FORMAT_VERSION = 1


@dataclass(frozen=True)
class ImageShardConversionStats:
    image_count: int
    shard_count: int
    skipped_count: int
    manifest_path: Path


def load_image_shard(path: str | Path, *, map_location: str | torch.device = "cpu") -> torch.Tensor:
    """Load one shard as a uint8 tensor shaped ``[N, 1, H, W]``."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    images = payload["images"]
    if not isinstance(images, torch.Tensor):
        raise TypeError("image shard payload does not contain a tensor under 'images'")
    return images


def load_image_shard_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest.get("format") != SHARD_FORMAT:
        raise ValueError(f"unsupported shard format: {manifest.get('format')!r}")
    if manifest.get("format_version") != SHARD_FORMAT_VERSION:
        raise ValueError(f"unsupported shard format version: {manifest.get('format_version')!r}")
    return manifest


def iter_image_shards(
    manifest_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    as_float: bool = False,
) -> Iterator[torch.Tensor]:
    """Yield tensors from all shards listed in a manifest."""

    manifest_path = Path(manifest_path)
    manifest = load_image_shard_manifest(manifest_path)
    for shard in manifest["shards"]:
        images = load_image_shard(manifest_path.parent / shard["file"], map_location=map_location)
        if as_float:
            images = images.float().div_(255.0)
        yield images


class GrayscaleImageShardDataset(torch.utils.data.Dataset):
    """Lazy map-style dataset backed by grayscale image shard files."""

    def __init__(self, manifest_path: str | Path, *, as_float: bool = False) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest = load_image_shard_manifest(self.manifest_path)
        self.as_float = as_float
        self.shards = self.manifest["shards"]
        counts = [int(shard["count"]) for shard in self.shards]
        total = 0
        self._offsets = [0]
        for count in counts:
            total += count
            self._offsets.append(total)
        self._cached_shard_index: int | None = None
        self._cached_images: torch.Tensor | None = None

    def __len__(self) -> int:
        return self._offsets[-1]

    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        shard_index = bisect.bisect_right(self._offsets, index) - 1
        local_index = index - self._offsets[shard_index]
        images = self._load_cached_shard(shard_index)
        image = images[local_index]
        if self.as_float:
            return image.float().div(255.0)
        return image

    def _load_cached_shard(self, shard_index: int) -> torch.Tensor:
        if self._cached_shard_index == shard_index and self._cached_images is not None:
            return self._cached_images
        shard = self.shards[shard_index]
        self._cached_images = load_image_shard(self.manifest_path.parent / shard["file"])
        self._cached_shard_index = shard_index
        return self._cached_images


class InMemoryGrayscaleImageShardDataset(torch.utils.data.Dataset):
    """Map-style dataset that preloads all grayscale shards as one uint8 tensor."""

    def __init__(self, manifest_path: str | Path, *, as_float: bool = False) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest = load_image_shard_manifest(self.manifest_path)
        self.as_float = as_float
        shards = [load_image_shard(self.manifest_path.parent / shard["file"]) for shard in self.manifest["shards"]]
        if not shards:
            raise ValueError(f"manifest contains no shards: {self.manifest_path}")
        self.images = torch.cat(shards, dim=0).contiguous()
        expected_count = int(self.manifest["image_count"])
        if self.images.shape[0] != expected_count:
            raise ValueError(f"manifest image_count={expected_count}, loaded {self.images.shape[0]} images")

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        image = self.images[index]
        if self.as_float:
            return image.float().div(255.0)
        return image


def convert_png_tar_to_grayscale_shards(
    tar_path: str | Path,
    output_dir: str | Path,
    *,
    shard_size: int = 65_536,
    image_size: tuple[int, int] = (32, 32),
    limit: int | None = None,
    overwrite: bool = False,
    resize: bool = False,
    skip_errors: bool = False,
    progress_every: int = 50_000,
) -> ImageShardConversionStats:
    """Stream PNGs from a tar archive into grayscale uint8 torch shards."""

    if shard_size <= 0:
        raise ValueError("shard_size must be positive")

    from PIL import Image

    tar_path = Path(tar_path)
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for old_file in output_dir.glob("images_*.pt"):
            old_file.unlink()
        if manifest_path.exists():
            manifest_path.unlink()

    width, height = image_size
    current_shard = torch.empty((shard_size, 1, height, width), dtype=torch.uint8)
    current_count = 0
    image_count = 0
    skipped_count = 0
    shards: list[dict[str, Any]] = []

    def flush_shard() -> None:
        nonlocal current_count
        if current_count == 0:
            return
        shard_index = len(shards)
        shard_name = f"images_{shard_index:05d}.pt"
        shard_path = output_dir / shard_name
        images = current_shard[:current_count].clone()
        torch.save(
            {
                "format": SHARD_FORMAT,
                "format_version": SHARD_FORMAT_VERSION,
                "images": images,
            },
            shard_path,
        )
        shards.append(
            {
                "file": shard_name,
                "count": current_count,
                "shape": list(images.shape),
                "dtype": "uint8",
            }
        )
        current_count = 0

    with tarfile.open(tar_path, mode="r|*") as archive:
        for member in archive:
            if limit is not None and image_count >= limit:
                break
            if not member.isfile() or not member.name.lower().endswith(".png"):
                continue

            try:
                file_obj = archive.extractfile(member)
                if file_obj is None:
                    raise ValueError(f"could not extract {member.name}")
                with file_obj, Image.open(file_obj) as image:
                    grayscale = image.convert("L")
                    if grayscale.size != (width, height):
                        if not resize:
                            raise ValueError(
                                f"expected image size {(width, height)}, got {grayscale.size} for {member.name}"
                            )
                        grayscale = grayscale.resize((width, height), Image.Resampling.BILINEAR)
                    pixels = torch.frombuffer(bytearray(grayscale.tobytes()), dtype=torch.uint8)
                    current_shard[current_count].copy_(pixels.reshape(1, height, width))
            except Exception:
                if not skip_errors:
                    raise
                skipped_count += 1
                continue

            current_count += 1
            image_count += 1
            if current_count == shard_size:
                flush_shard()
            if progress_every > 0 and image_count % progress_every == 0:
                print(f"converted {image_count} images into {len(shards)} complete shards", flush=True)

    flush_shard()
    manifest = {
        "format": SHARD_FORMAT,
        "format_version": SHARD_FORMAT_VERSION,
        "source_tar": str(tar_path),
        "image_count": image_count,
        "skipped_count": skipped_count,
        "image_size": [width, height],
        "channels": 1,
        "dtype": "uint8",
        "tensor_shape": ["N", 1, height, width],
        "shard_size": shard_size,
        "shards": shards,
    }
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")

    return ImageShardConversionStats(
        image_count=image_count,
        shard_count=len(shards),
        skipped_count=skipped_count,
        manifest_path=manifest_path,
    )