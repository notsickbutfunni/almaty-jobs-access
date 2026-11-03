"""Logging setup shared by the CLI and pipeline modules."""

from __future__ import annotations

import logging

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
DATE_FORMAT = "%H:%M:%S"


def setup_logging(verbose: bool = False) -> None:
    """Configure root logging once per CLI invocation.

    Pipeline modules get their own logger with ``logging.getLogger(__name__)``
    and never call ``basicConfig`` themselves, so logging stays configured in
    exactly one place.

    Parameters
    ----------
    verbose:
        If True, show DEBUG messages; otherwise INFO and above.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=DATE_FORMAT)
