import logging
import sys


def configure_logs(**kwargs):
    """Helper method for easily configuring logs from the python shell.
    Args:
        level: INFO, DEBUG, ERROR, WARNING, CRITICAL
        format: str
        filename: str/Path
        filemode: str
    """
    level_str = kwargs.get("level", "INFO")
    level = getattr(logging, level_str)
    log_format = kwargs.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    if kwargs.get("filename"):
        logging.basicConfig(filename=kwargs.get("filename"), filemode=kwargs.get("filemode", "a"),
                            format=log_format)
    else:
        logging.basicConfig(stream=sys.stdout, format=log_format)
    package_logger = logging.getLogger(__package__)
    package_logger.setLevel(level)
