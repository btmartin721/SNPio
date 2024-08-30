import logging
import sys

def setup_logger(name: str, log_file: str = "snpio.log", level=logging.INFO) -> logging.Logger:
    """
    Setup a centralized logger that outputs logs to both stdout and a single file.

    Args:
        name (str): The name of the module from where the logger is being called.
        log_file (str): The name of the log file where all logs will be stored.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: Configured centralized logger instance.
    """
    
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if logger already has handlers (to prevent duplicate logs)
    if not logger.handlers:
        # StreamHandler for stdout
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # FileHandler for logging to a file
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
