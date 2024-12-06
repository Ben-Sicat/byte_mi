import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

def setup_logging(
    log_dir: str = "logs",
    log_level: int = logging.INFO,
    log_prefix: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration to write to both file and console.
    
    Args:
        log_dir: Directory to store log files
        log_level: Logging level (default: INFO)
        log_prefix: Optional prefix for log filename
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    prefix = f"{log_prefix}_" if log_prefix else ""
    log_file = log_path / f"{prefix}log_{timestamp}.txt"
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplication
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Log initial message
    root_logger.info(f"Logging initialized. Log file: {log_file}")
    
    return root_logger
