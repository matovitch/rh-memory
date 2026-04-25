from __future__ import annotations

from typing import Callable, Iterable, Iterator, TypeVar

from torch.utils.data import IterableDataset

T = TypeVar("T")
U = TypeVar("U")


def iter_map(stream: Iterable[T], fn: Callable[[T], U]) -> Iterator[U]:
    for item in stream:
        yield fn(item)


def iter_take(stream: Iterable[T], max_items: int | None) -> Iterator[T]:
    if max_items is None:
        yield from stream
        return
    n = 0
    for item in stream:
        if n >= max_items:
            break
        yield item
        n += 1


def worker_init_fn(worker_id):
    import torch

    worker_seed = torch.initial_seed() % 2**32
    torch.manual_seed(worker_seed)


class IterableFactoryDataset(IterableDataset):
    """Lightweight dataset wrapper around an iterable factory."""

    def __init__(self, factory: Callable[[], Iterable]):
        super().__init__()
        self.factory = factory

    def __iter__(self):
        yield from self.factory()
