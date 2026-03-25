"""Shared utilities: logging, retry, path helpers."""

from __future__ import annotations

import functools
import logging
import time
from pathlib import Path
from typing import Tuple, Type


def setup_logging(log_level: str, log_file: Path) -> logging.Logger:
    """Configure root logger with file + console handlers."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("btcdump")
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if not root.handlers:
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        ch.setFormatter(fmt)
        root.addHandler(ch)

    return root


def retry(
    max_retries: int = 3,
    backoff: float = 1.5,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
):
    """Decorator: exponential-backoff retry for flaky calls."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_retries - 1:
                        raise
                    wait = backoff ** attempt
                    logging.getLogger(__name__).warning(
                        "Retry %d/%d for %s after %.1fs: %s",
                        attempt + 1, max_retries, func.__name__, wait, exc,
                    )
                    time.sleep(wait)
        return wrapper

    return decorator


def ensure_dirs(*paths: Path) -> None:
    """Create directories if they don't exist."""
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
