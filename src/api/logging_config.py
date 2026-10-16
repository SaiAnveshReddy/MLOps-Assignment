"""Structured JSON request logging (Task 8 - Monitoring & Logging).

A plain `logging.StreamHandler` with a JSON formatter is enough for this
assignment's scope: logs go to stdout, which is exactly what `docker logs`
and a Kubernetes pod's log stream capture without any extra plumbing.
"""

import logging
import sys

from pythonjsonlogger import jsonlogger

LOGGER_NAME = "cardiorisk_api"


def get_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
