import asyncio
from pathlib import Path
from typing import Iterable, TypeVar, Union, Iterator, Tuple, Awaitable

PROJECT_ROOT = Path(__file__).parent.parent.parent

T = TypeVar("T")


def iterate(iterable: Iterable[T], start: int = 0, enumerate_items: bool = False) -> Iterator[Union[T, Tuple[int, T]]]:
    """
    Creates an iterator that uses tqdm if available, otherwise falls back to regular iteration.
    Supports optional enumeration of items.

    Args:
        iterable: The iterable to loop over
        start: Starting index for enumeration (default: 0)
        enumerate_items: Whether to enumerate the items (default: False)

    Returns:
        An iterator that yields either items or (index, item) pairs if enumerate_items is True
    """
    try:
        from tqdm import tqdm

        iterator = tqdm(iterable)
    except ImportError:
        iterator = iterable

    if enumerate_items:
        return enumerate(iterator, start=start)
    return iterator


async def asyncio_gather(*tasks: Awaitable[T], return_exceptions: bool = False) -> list[T]:
    """
    Gathers awaitables with a progress bar when tqdm is available,
    otherwise falls back to regular asyncio.gather.

    Args:
        *tasks: Awaitable objects to gather
        return_exceptions: If True, exceptions are returned rather than raised

    Returns:
        List of results from the gathered tasks
    """
    try:
        from tqdm.asyncio import tqdm_asyncio

        return await tqdm_asyncio.gather(*tasks)
    except ImportError:
        return await asyncio.gather(*tasks, return_exceptions=return_exceptions)
