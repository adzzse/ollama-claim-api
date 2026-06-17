import logging
import os

from app.env import load_runtime_env


def configure_app_logging() -> None:
    load_runtime_env()
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger = logging.getLogger("app")
    logger.setLevel(level)
    logger.propagate = True

    if logger.handlers:
        for handler in logger.handlers:
            handler.setLevel(logging.NOTSET)
        return

    handler = logging.StreamHandler()
    handler.setLevel(logging.NOTSET)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
    logger.addHandler(handler)
