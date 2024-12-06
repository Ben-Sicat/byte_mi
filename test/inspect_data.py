import sys
import os
from pathlib import Path
import numpy as np
import cv2
import json
from typing import Dict
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_raw_file(file_path: Path) -> Dict:
    """Analyze raw depth file"""
    try:
        # Try different data types
        data_types = [np.float32, np.uint16, np.uint8]
        results = {}
        
        for dtype in data_types:
            try:
                raw_data = np.fromfile(file_path, dtype=dtype)
                results[dtype.__name__] = {
                    'size': raw_data.size,
                    'min': float(raw_data.min()),
                    'max': float(raw_data.max()),
                    'possible_shapes': [
                        f"{s} x {raw_data.size // s}" 
                        for s in range(1, raw_data.size + 1) 
                        if raw_data.size % s == 0
                    ][:5]  # Show first 5 possible shapes
                }
            except Exception as e:
                logger.warning(f"Could not read as {dtype.__name__}: {str(e)}")
                continue
        
        return results
    except Exception as e:
        logger.error(f"Error analyzing raw file: {str(e)}")
        return {}

def inspect_rgb_file(file_path: Path) -> Dict:
    """Analyze RGB image file"""
    try:
        img = cv2.imread(str(file_path))
        if img is None:
            raise ValueError("Could not read image file")
            
        return {
            'shape': img.shape,
            'dtype': img.dtype,
            'min_val': float(img.min()),
            'max_val': float(img.max())
        }
    except Exception as e:
        logger.error(f"Error analyzing RGB file: {str(e)}")
        return {}

def inspect_coco_file(file_path: Path) -> Dict:
    """Analyze COCO annotation file"""
    try:
        with open(file_path, 'r') as f:
            coco_data = json.load(f)
            
        return {
            'num_images': len(coco_data.get('images', [])),
            'num_annotations': len(coco_data.get('annotations', [])),
            'categories': [cat['name'] for cat in coco_data.get('categories', [])],
            'image_shapes': [
                (img['height'], img['width']) 
                for img in coco_data.get('images', [])
            ]
        }
    except Exception as e:
        logger.error(f"Error analyzing COCO file: {str(e)}")
        return {}

def main():
    try:
        data_dir = Path("data")
        
        # Inspect first raw file
        raw_files = list((data_dir / "rgbd").glob("*.raw"))
        if raw_files:
            logger.info("\nAnalyzing RAW file:")
            raw_results = inspect_raw_file(raw_files[0])
            logger.info(f"File: {raw_files[0].name}")
            for dtype, info in raw_results.items():
                logger.info(f"\nAs {dtype}:")
                logger.info(f"Size: {info['size']}")
                logger.info(f"Range: {info['min']} to {info['max']}")
                logger.info(f"Possible dimensions: {info['possible_shapes']}")
        else:
            logger.warning("No .raw files found")
        
        # Inspect first PNG file
        png_files = list((data_dir / "segmented").glob("*.png"))
        if png_files:
            logger.info("\nAnalyzing RGB file:")
            rgb_results = inspect_rgb_file(png_files[0])
            logger.info(f"File: {png_files[0].name}")
            logger.info(f"Shape: {rgb_results.get('shape')}")
            logger.info(f"Data type: {rgb_results.get('dtype')}")
            logger.info(f"Value range: {rgb_results.get('min_val')} to {rgb_results.get('max_val')}")
        else:
            logger.warning("No .png files found")
        
        # Inspect COCO file
        coco_file = data_dir / "segmented" / "_annotations.coco.json"
        if coco_file.exists():
            logger.info("\nAnalyzing COCO file:")
            coco_results = inspect_coco_file(coco_file)
            logger.info(f"Number of images: {coco_results.get('num_images')}")
            logger.info(f"Number of annotations: {coco_results.get('num_annotations')}")
            logger.info(f"Categories: {coco_results.get('categories')}")
            logger.info(f"Image shapes: {coco_results.get('image_shapes')}")
        else:
            logger.warning("COCO annotation file not found")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()
