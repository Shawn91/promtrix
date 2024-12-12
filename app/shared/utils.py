import asyncio
import logging
from typing import Iterable, TypeVar, Union, Iterator, Tuple, Awaitable, Optional

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


class LoggerConfig:
    _instance: Optional["LoggerConfig"] = None
    _logger: Optional[logging.Logger] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._logger is None:
            # Create logger
            self._logger = logging.getLogger("Promtrix")
            self._logger.setLevel(logging.INFO)

            # Remove existing handlers to avoid duplicates
            if self._logger.handlers:
                self._logger.handlers.clear()

            # Create formatter
            formatter = logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s")

            # Console handler only
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)

    @property
    def logger(self) -> logging.Logger:
        return self._logger


logger = LoggerConfig().logger
