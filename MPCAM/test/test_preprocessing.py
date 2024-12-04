import sys
import os
from pathlib import Path
import logging
import json

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.preprocessing import PreprocessingPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_first_files():
    try:
        # Get first .raw file from rgbd directory
        rgbd_dir = Path("data/rgbd")
        raw_files = list(rgbd_dir.glob("*.raw"))
        if not raw_files:
            raise FileNotFoundError("No .raw files found in rgbd directory")
        raw_file = raw_files[0]
        
        # Extract frame_id from filename
        frame_id = raw_file.stem.replace("depth_frame_", "")
        
        # Get corresponding files
        segmented_dir = Path("data/segmented")
        coco_file = next(segmented_dir.glob("*_annotations.coco.json"))
        png_file = next(segmented_dir.glob(f"rgb_frame_{frame_id}.png"))
        
        logger.info(f"Found files:")
        logger.info(f"RAW: {raw_file.name}")
        logger.info(f"COCO: {coco_file.name}")
        logger.info(f"PNG: {png_file.name}")
        
        return {
            'frame_id': frame_id,
            'raw_path': str(raw_file),
            'coco_path': str(coco_file),
            'png_path': str(png_file)
        }
        
    except Exception as e:
        logger.error(f"Error finding files: {str(e)}")
        raise
def create_test_config(files):
    config = {
        "data_dir": str(Path("data")),
        "output_dir": str(Path("data/upscaled")),
        "coco_file": files['coco_path'],
        "frame_ids": [files['frame_id']],
        "camera_height": 33.0,
        "plate_diameter": 25.5,
        "plate_height": 0.7
    }
    
    config_path = Path("test_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    return str(config_path)
def test_preprocessing():
    try:
        files = get_first_files()
        
        config_path = create_test_config(files)
        logger.info(f"Created config file: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        pipeline = PreprocessingPipeline(config)
        
        logger.info("Testing data loading...")
        data = pipeline.load_data(files['frame_id'])
        logger.info("Data loading successful")
        
        logger.info("Testing full preprocessing pipeline...")
        result = pipeline.process_single_image(files['frame_id'])
        logger.info("Preprocessing completed successfully")
        
        upscaled_dir = Path("data/upscaled")
        if upscaled_dir.exists():
            output_files = list(upscaled_dir.glob("*"))
            logger.info(f"Files generated in upscaled directory: {[f.name for f in output_files]}")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_preprocessing()
    if success:
        print("All tests passed!")
    else:
        print("Tests failed!")
