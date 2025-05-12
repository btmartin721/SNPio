import logging
import sys
from pathlib import Path


class LoggerManager:
    """A modular and fully-featured logger class for managing logging across modules.

    This class sets up a logger with the specified name, log file, and logging level. It supports logging to both stdout and a file, with customizable formats and date formats.

    Example:
        >>> logger_manager = LoggerManager(name="my_module", prefix="my_prefix", debug=True)
        >>> logger = logger_manager.get_logger("my_module")
        >>>
        >>> logger.info("This is an info message.")
        >>> logger.debug("This is a debug message.")
        >>> logger.warning("This is a warning message.")
        >>> logger.error("This is an error message.")
        >>> logger.critical("This is a critical message.")
        >>> logger_manager.set_level(logging.WARNING)
        >>> logger_manager.remove_handler(logger)

    Attributes:
        name (str): The name of the logger.
        logger (logging.Logger): The configured logger instance.

    Methods:
        get_logger(): Returns the logger instance.
        set_level(level): Sets the logging level.
        add_handler(handler): Adds a logging handler.
        remove_handler(handler): Removes a logging handler.
    """

    def __init__(
        self,
        name: str,
        prefix: str | None = "",
        debug: bool = False,
        verbose: bool = True,
        log_file: str | Path | None = None,
        level: int | None = None,
        to_console: bool = True,
        to_file: bool = True,
        log_format: str = "%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s",
        date_format: str = "%Y-%m-%d %H:%M:%S",
    ):
        """Initializes the LoggerManager.

        This class sets up a logger with the specified name, log file, and logging level. It supports logging to both stdout and a file, with customizable formats and date formats. The default log format is "%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s", and the default date format is "%Y-%m-%d %H:%M:%S".

        Args:
            name (str): Name of the logger.
            prefix (str, optional): Prefix for the log file directory.
            debug (bool, optional): If True, sets log level to DEBUG.
            verbose (bool, optional): If False, suppresses INFO and WARNING messages.
            log_file (str or Path, optional): Path to the log file.
            level (int, optional): Explicit logging level (overrides debug and verbose).
            to_console (bool, optional): Whether to log to console. Defaults to True.
            to_file (bool, optional): Whether to log to file. Defaults to True.
            log_format (str, optional): Format of the log messages.
            date_format (str, optional): Format of the date in log messages.
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.verbose = verbose

        # Set default log level to INFO
        self.logger.setLevel(logging.INFO)

        # Determine log level
        if level is not None:
            self.logger.setLevel(level)
        elif debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

        # Check if handlers already exist to prevent duplicates
        if not self.logger.handlers:
            handlers = []

            if to_console:
                # Add StreamHandler to stdout
                stream_handler = logging.StreamHandler(sys.stdout)
                stream_handler.setFormatter(formatter)
                handlers.append(stream_handler)

            if to_file:
                # Determine log_file path
                if log_file:
                    log_file = Path(log_file)
                elif prefix:
                    # Construct log_file path using prefix
                    log_file = Path(f"{prefix}_output") / "logs" / f"{name}.log"
                else:
                    log_file = Path("logs") / f"{name}.log"

                # Ensure parent directories exist
                log_file.parent.mkdir(exist_ok=True, parents=True)

                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                handlers.append(file_handler)

            # Apply verbose filtering if necessary
            for handler in handlers:
                if not verbose:
                    # Set handler level to ERROR to suppress INFO and WARNING
                    handler.setLevel(logging.ERROR)
                else:
                    # Ensure handler level is not set (inherits logger level)
                    handler.setLevel(logging.NOTSET)
                self.logger.addHandler(handler)

    def get_logger(self) -> logging.Logger:
        """Returns the logger instance.

        Returns:
            logging.Logger: The configured logger.
        """
        return self.logger

    def set_level(self, level: str) -> None:
        """Sets the logging level.

        Args:
            level (str): The new logging level.
        """
        loglevel: int = logging.NOTSET

        if self.verbose:
            if level == "DEBUG":
                loglevel = logging.DEBUG
            elif level == "INFO":
                loglevel = logging.INFO
            elif level == "WARNING":
                loglevel = logging.WARNING
            elif level == "ERROR":
                loglevel = logging.ERROR
            elif level == "CRITICAL":
                loglevel = logging.CRITICAL
            else:
                raise ValueError(
                    "Invalid logging level. Choose from DEBUG, INFO, WARNING, ERROR, CRITICAL."
                )

        else:
            if level == "DEBUG":
                loglevel = logging.DEBUG
            elif level == "INFO":
                loglevel = logging.ERROR
            elif level == "WARNING":
                loglevel = logging.ERROR
            elif level == "ERROR":
                loglevel = logging.ERROR
            elif level == "CRITICAL":
                loglevel = logging.CRITICAL
            else:
                raise ValueError(
                    "Invalid logging level. Choose from DEBUG, INFO, WARNING, ERROR, CRITICAL."
                )

        self.logger.setLevel(loglevel)

    def add_handler(self, handler: logging.Handler) -> None:
        """Adds a logging handler if it's not already added.

        Args:
            handler (logging.Handler): The handler to add.
        """
        if handler not in self.logger.handlers:
            self.logger.addHandler(handler)

    def remove_handler(self, handler: logging.Handler) -> None:
        """Removes a logging handler if it exists.

        Args:
            handler (logging.Handler): The handler to remove.
        """
        if handler in self.logger.handlers:
            self.logger.removeHandler(handler)

    def change_name(self, new_name: str) -> None:
        """Changes the name of the logger.

        Args:
            new_name (str): The new name for the logger.
        """
        self.logger.name = new_name
        self.name = new_name
