"""Structured logging setup used across the project.

A single :func:`get_logger` helper returns a ``logging.Logger`` wired
up with a ``rich`` handler so that INFO-and-above messages render with
colour and tracebacks are readable in the terminal.
"""

from __future__ import annotations

import logging

from rich.logging import RichHandler


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger configured with a Rich handler.

    Args:
        name: Logger name — conventionally the calling module's ``__name__``.
        level: Minimum log level to emit. Defaults to ``logging.INFO``.

    Returns:
        A configured :class:`logging.Logger`. Repeated calls with the same
        ``name`` return the same instance without duplicating handlers.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
            markup=False,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    return logger
