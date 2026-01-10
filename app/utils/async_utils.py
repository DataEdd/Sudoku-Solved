"""
Async utilities for running blocking operations in thread executors.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from typing import Callable, TypeVar

T = TypeVar("T")

# Shared executor for CPU-bound tasks
_executor = ThreadPoolExecutor(max_workers=4)


def run_sync(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to run a synchronous function in a thread executor.

    Usage:
        @run_sync
        def blocking_operation():
            # CPU-intensive work
            pass

        # Call as async
        result = await blocking_operation()
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, lambda: func(*args, **kwargs))

    return wrapper


async def run_in_executor(func: Callable[..., T], *args, **kwargs) -> T:
    """
    Run a blocking function in the thread pool executor.

    Usage:
        result = await run_in_executor(blocking_func, arg1, arg2)
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, lambda: func(*args, **kwargs))
