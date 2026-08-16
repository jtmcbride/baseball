from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def setup_logging(level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    from bbcore.config import get_settings

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)-24s %(message)s", "%H:%M:%S")
    )
    root = logging.getLogger()
    root.setLevel(level or get_settings().log_level)
    root.handlers = [handler]
    # These are chatty at DEBUG and drown out ingest progress.
    for noisy in ("urllib3", "httpx", "httpcore", "requests"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
