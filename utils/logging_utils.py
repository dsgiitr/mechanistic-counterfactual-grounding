"""Logging utilities for the training script."""

import logging
from pathlib import Path


def setup_logging(output_dir: str) -> None:
    """Configure root logging to both console and a log file."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    log_file = out / "train.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Prevent duplicate handlers when setup is called multiple times.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    root.addHandler(stream_handler)
    root.addHandler(file_handler)
