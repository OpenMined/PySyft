"""Configure handlers and formats for application loggers."""
# stdlib
import configparser
import logging
from pprint import pformat
import sys
from typing import Dict

# third party
from loguru import logger

LOGGER_CONFIG_FILE = "app/logger/config.ini"
LOGGER_CONFIG_SECTION = "logger"


class LogHandler:
    def __init__(self) -> None:
        self.config = {
            "format": (
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>: "
                "<level>{message}</level>"
            ),
            "level": 0,
            "sink": sys.stdout,
            "filter": None,
            "compression": None,
            "rotation": None,
            "retention": None,
        }

        try:
            config = configparser.ConfigParser()
            config_file = open(LOGGER_CONFIG_FILE)
            config.read_file(config_file)

            for option in config.options(LOGGER_CONFIG_SECTION):
                # Review if it is better to "explode" and switch loading all settings
                # properties on a per-option basis (re:mypy)
                self.config[option] = config.get(LOGGER_CONFIG_SECTION, option)
        except IOError:
            logger.debug(f"Failed to find the log config file at {LOGGER_CONFIG_FILE}.")
        except configparser.NoSectionError:
            logger.debug(
                f"Failed to find section {LOGGER_CONFIG_SECTION} in the log config file."
            )
        except Exception as err:
            logger.debug(f"Failed loading log configs. {err}")
        finally:
            logger.debug(
                f"Starting log settings:\n{pformat(self.config, indent=2, compact=False, width=88)}"
            )

    # Review mypy here
    def get_initial_config(self) -> Dict[str, object]:
        """
        Returns the configuration used to setup the logs
        """
        return self.config

    def format_record(self, record: dict) -> str:
        """
        Custom logger format for loguru.
        """
        format_string: str = str(self.config["format"])

        if record["extra"] is not None:
            for key in record["extra"].keys():
                record["extra"][key] = pformat(
                    record["extra"][key], indent=2, compact=False, width=88
                )
                format_string += "\n{extra[" + key + "]}"

        format_string += "<level>{exception}</level>\n"

        return format_string

    def init(self) -> None:
        """
        Redirects uvicorn and fastapi loggers to a loguru sink. Call init on startup.
        """
        intercept_handler = InterceptHandler()

        for log in [
            "uvicorn",
            "uvicorn.error",
            "uvicorn.access",
            "fastapi",
            "app",
        ]:
            logging.getLogger(log).handlers = [intercept_handler]

        logger.configure(
            handlers=[{"sink": sys.stdout, "level": 0, "format": self.format_record}],
        )

        try:
            if self.config["sink"] is not sys.stdout:
                logger.add(**self.config)
                logger.debug(f"Logging to {self.config['sink']}")
        except Exception as err:
            logger.debug(f"Failed creating a new sink. Check your log config. {err}")


class InterceptHandler(logging.Handler):
    """
    Check https://loguru.readthedocs.io/en/stable/overview.html#entirely-compatible-with-standard-logging
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno # type: ignore

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:  # type: ignore
            frame = frame.f_back  # type: ignore
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )
