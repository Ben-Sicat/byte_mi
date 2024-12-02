import sys
from pathlib import Path
import os
import json  # Add required imports for 3d_reconstruction.py
import numpy as np
import cv2
import plotly.graph_objects as go
from scipy.stats import mode
import traceback
import requests

def run_pipeline():
    try:
        # Add src directory to Python path
        src_path = Path(__file__).parent
        sys.path.append(str(src_path))
        
        # Run preprocessing pipeline
        print("\nStarting preprocessing pipeline...")
        from preprocessing.test import test_preprocessing_pipeline
        test_preprocessing_pipeline()
        
        # Run 3D reconstruction by executing the script with all required imports
        print("\nStarting 3D reconstruction...")
        reconstruction_path = os.path.join(src_path, "reconstruction", "3d_reconstruction.py")
        
        # Create a namespace with all required imports
        namespace = {
            'json': json,
            'np': np,
            'cv2': cv2,
            'go': go,
            'mode': mode,
            'traceback': traceback,
            'os': os,
            'requests': requests,
            '__name__': '__main__'  # This ensures the if __name__ == '__main__' block runs
        }
        
        exec(open(reconstruction_path).read(), namespace)
        
        print("\nPipeline completed!")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()
