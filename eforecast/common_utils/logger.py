import logging
import os


def create_logger(logger_name, abs_path, logger_path, write_type='w'):
    assert write_type in ('w', 'a')
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(os.path.join(abs_path, logger_path), write_type)
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)
    return logger
