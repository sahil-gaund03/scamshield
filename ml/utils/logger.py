import logging
import sys
from logging.handlers import RotatingFileHandler
from ml.config.config import LOG_FILE_PATH

def setup_logger(name: str = "scamshield") -> logging.Logger:
    """
    Sets up a logger that outputs to both standard output and a rotating log file.
    """
    logger = logging.getLogger(name)
    
    # If logger is already configured, don't add duplicate handlers
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Rotating File Handler (10MB limit per file, keeping up to 5 files)
    try:
        file_handler = RotatingFileHandler(
            LOG_FILE_PATH,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not configure file logger at {LOG_FILE_PATH}: {e}")

    return logger

# Shared default logger instance
logger = setup_logger()
