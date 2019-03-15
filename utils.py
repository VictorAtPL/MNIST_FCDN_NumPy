import logging

from consts import APP_NAME


def setup_logger():
    logger = get_logger()
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    logger.addHandler(ch)


def get_logger():
    return logging.getLogger(APP_NAME)
