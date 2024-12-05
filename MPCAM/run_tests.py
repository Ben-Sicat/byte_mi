import os
import sys
from pathlib import Path
import pytest
import logging

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.logging_utils import setup_logging

def main():
    # Set up logging
    logger = setup_logging(
        log_dir=project_root / "logs/tests",
        log_prefix="test"
    )
    
    try:
        logger.info("Starting test suite execution")
        
        # Run preprocessing tests first
        logger.info("\nRunning preprocessing tests...")
        preprocessing_result = pytest.main([
            str(project_root / 'test/test_preprocessing.py'),
            '-v',
            '--capture=tee-sys'
        ])
        
        if preprocessing_result == 0:
            logger.info("\nPreprocessing tests passed. Running reconstruction tests...")
            # Run point cloud and reconstruction tests
            reconstruction_result = pytest.main([
                str(project_root / 'test/test_point_cloud.py'),
                str(project_root / 'test/test_reconstruction.py'),
                '-v',
                '--capture=tee-sys'
            ])
            
            if reconstruction_result == 0:
                logger.info("\nAll tests passed successfully!")
            else:
                logger.error("\nReconstruction tests failed!")
        else:
            logger.error("\nPreprocessing tests failed!")
            
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}", exc_info=True)
        return 1
        
    finally:
        logger.info("Test suite execution completed")

if __name__ == "__main__":
    main()
