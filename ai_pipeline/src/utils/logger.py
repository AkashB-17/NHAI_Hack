import sys

from loguru import logger


def setup_logger(level: str = "INFO") -> None:
    logger.remove()
    logger.add(sys.stderr, level=level)
