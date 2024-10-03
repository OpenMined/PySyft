def make_nb_logger(name="notebook"):
    # stdlib
    import logging

    logger = logging.getLogger()
    file_logging = logging.FileHandler(f"{name}.log")
    file_logging.setFormatter(
        logging.Formatter(
            "%(asctime)s - pid-%(process)d tid-%(thread)d - %(levelname)s - %(name)s - %(message)s"
        )
    )
    file_logging.setLevel(logging.INFO)

    logger.addHandler(file_logging)
    logger.setLevel(logging.INFO)
    return logger
