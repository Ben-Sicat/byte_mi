import os
import sys
from pathlib import Path
import argparse
import logging

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.preprocessing.preprocessing import run_preprocessing
from src.utils.logging_utils import setup_logging

def main():
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline")
    parser.add_argument('--config', required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(
        log_dir=project_root / "logs/preprocessing",
        log_prefix="preprocess"
    )
    
    try:
        logger.info(f"Starting preprocessing pipeline with config: {args.config}")
        run_preprocessing(args.config)
        logger.info("Preprocessing pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    main()
