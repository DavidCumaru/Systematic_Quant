"""
logging.py
==========
Centralized logging setup for the Systematic Alpha pipeline.

Usage
-----
    from autotrader.utils.logging import setup_logging
    setup_logging(verbose=False)               # basic setup
    setup_logging(log_file="logs/run.log")     # with file handler
"""

import logging
import sys
from pathlib import Path


def setup_logging(
    verbose: bool = False,
    log_file: str | Path | None = None,
) -> None:
    """
    Configure root logger with console + optional file handler.

    Parameters
    ----------
    verbose  : if True, sets level to DEBUG; otherwise INFO
    log_file : optional path for a rotating file handler
    """
    level = logging.DEBUG if verbose else logging.INFO
    fmt   = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [
        logging.StreamHandler(
            open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
        )
    ]

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)
