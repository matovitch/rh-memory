from __future__ import annotations

import io
import tarfile

import torch
from PIL import Image

from rh_memory.image_shards import (
    GrayscaleImageShardDataset,
    InMemoryGrayscaleImageShardDataset,
    convert_png_tar_to_grayscale_shards,
    iter_image_shards,
    load_image_shard_manifest,
)


def _write_png_tar(path, image_count: int = 3) -> None:
    with tarfile.open(path, "w") as archive:
        for index in range(image_count):
            image = Image.new("RGB", (32, 32), color=(index * 40, index * 20, index * 10))
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            payload = buffer.getvalue()
            info = tarfile.TarInfo(f"images/{index:03d}.png")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def test_convert_png_tar_to_grayscale_shards(tmp_path) -> None:
    tar_path = tmp_path / "images.tar"
    output_dir = tmp_path / "shards"
    _write_png_tar(tar_path, image_count=3)

    stats = convert_png_tar_to_grayscale_shards(
        tar_path,
        output_dir,
        shard_size=2,
        progress_every=0,
    )

    assert stats.image_count == 3
    assert stats.shard_count == 2
    manifest = load_image_shard_manifest(stats.manifest_path)
    assert manifest["image_count"] == 3
    shards = list(iter_image_shards(stats.manifest_path))
    assert [tuple(shard.shape) for shard in shards] == [(2, 1, 32, 32), (1, 1, 32, 32)]
    assert shards[0].dtype == torch.uint8


def test_grayscale_image_shard_dataset(tmp_path) -> None:
    tar_path = tmp_path / "images.tar"
    output_dir = tmp_path / "shards"
    _write_png_tar(tar_path, image_count=3)
    stats = convert_png_tar_to_grayscale_shards(
        tar_path,
        output_dir,
        shard_size=2,
        progress_every=0,
    )

    dataset = GrayscaleImageShardDataset(stats.manifest_path, as_float=True)

    assert len(dataset) == 3
    assert tuple(dataset[2].shape) == (1, 32, 32)
    assert dataset[2].dtype == torch.float32
    assert float(dataset[2].max()) <= 1.0


def test_in_memory_grayscale_image_shard_dataset(tmp_path) -> None:
    tar_path = tmp_path / "images.tar"
    output_dir = tmp_path / "shards"
    _write_png_tar(tar_path, image_count=3)
    stats = convert_png_tar_to_grayscale_shards(
        tar_path,
        output_dir,
        shard_size=2,
        progress_every=0,
    )

    dataset = InMemoryGrayscaleImageShardDataset(stats.manifest_path, as_float=True)

    assert len(dataset) == 3
    assert tuple(dataset.images.shape) == (3, 1, 32, 32)
    assert dataset.images.dtype == torch.uint8
    assert tuple(dataset[1].shape) == (1, 32, 32)
    assert dataset[1].dtype == torch.float32