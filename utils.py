import logging

from consts import APP_NAME


def setup_logger(logging_level: str = logging.INFO):
    logger = get_logger()

    # Clear previous handlers if setup_logger invoked again
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.setLevel(logging_level)

    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)

    logger.addHandler(ch)


def get_logger():
    return logging.getLogger(APP_NAME)
