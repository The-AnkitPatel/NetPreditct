import logging
import sys


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configures structured format logger for NetPredict backend."""
    logger = logging.getLogger("netpredict")
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = setup_logging()
