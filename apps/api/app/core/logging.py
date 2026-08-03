import sys

from loguru import logger


def configure_logging(log_level: str) -> None:
    """Configure structured application logging."""
    logger.remove()
    logger.add(sys.stderr, level=log_level.upper(), serialize=True, backtrace=False, diagnose=False)
