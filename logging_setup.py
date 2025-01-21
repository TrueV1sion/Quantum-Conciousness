# logging_setup.py

import logging
import sys
from pathlib import Path

def setup_logging(log_level: int = logging.DEBUG, log_file: str = None):
    """
    Configure the root logger with console and optional file output.
    
    Args:
        log_level: The logging level to use (default: DEBUG)
        log_file: Optional path to a log file. If provided, logs will be written to this file.
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear any existing handlers
    root_logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if log_file is specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Create logger instance for this module
    logger = logging.getLogger(__name__)
    logger.debug("Logging setup completed")
