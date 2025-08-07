from typing import TypeVar, Iterable, Callable, Generator

T = TypeVar('T')
K = TypeVar('K')

def dedup(items: Iterable[T], key: Callable[[T], K] = lambda x: x) -> Generator[T, None, None]:
    seen = set()
    for item in items:
        k = key(item)
        if k not in seen:
            seen.add(k)
            yield item
