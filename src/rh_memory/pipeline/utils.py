from __future__ import annotations

from typing import Callable, Iterable, Iterator, TypeVar

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