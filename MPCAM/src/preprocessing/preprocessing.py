import cv2
import numpy as np
from typing import Dict, Optional
import logging
from pathlib import Path
import json

from ..core.depth_processor import DepthProcessor
from ..core.image_alignment import ImageAligner
from .calibration import CameraCalibrator
from .noise_reduction import DepthNoiseReducer
from ..utils.coco_utils import CocoHandler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PreprocessingPipeline:
    def __init__(self, config: Dict):
        """
        Initialize preprocessing pipeline.
        
        Args:
            config: Dict containing:
                - data_dir: Path to data directory
                - output_dir: Path to save processed data
                - coco_file: Path to COCO annotations
                - rgbd_shape: Original RGBD shape (height, width)
                - rgb_shape: RGB reference shape (height, width)
                - camera_height: Height of camera in cm
                - plate_diameter: Diameter of plate in cm
                - plate_height: Height of plate in cm
        """
        self.config = config
        self.data_dir = Path(config['data_dir'])
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.depth_processor = DepthProcessor()
        self.image_aligner = ImageAligner(config['coco_file'])
        self.calibrator = CameraCalibrator()
        self.noise_reducer = DepthNoiseReducer()
        self.coco_handler = CocoHandler(config['coco_file'])
        
        # Set reference sizes
        self.image_aligner.set_reference_sizes(
            rgb_shape=config['rgb_shape'],
            rgbd_shape=(120, 120)  # Raw depth resolution
        )
        
        logger.info("Initialized preprocessing pipeline")
        
    def load_data(self, frame_id: str) -> Dict:
        """Load all necessary data for processing"""
        try:
            # Load RGB image
            rgb_path = self.data_dir / "segmented" / f"rgb_frame_{frame_id}.png"
            if not rgb_path.exists():
                raise FileNotFoundError(f"RGB image not found: {rgb_path}")
                
            rgb_image = cv2.imread(str(rgb_path))
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            
            # Load depth data
            depth_path = self.data_dir / "rgbd" / f"depth_frame_{frame_id}.raw"
            if not depth_path.exists():
                raise FileNotFoundError(f"Depth data not found: {depth_path}")
            
            # Load and process depth
            raw_depth = self.depth_processor.load_raw_rgbd(str(depth_path))
            processed_depth = self.depth_processor.process_depth(raw_depth)
            upscaled_depth = self.depth_processor.upscale_depth(processed_depth)
            
            # Get plate mask
            plate_mask = self.coco_handler.create_category_mask(
                frame_id, 
                'case',  # Using 'case' as category from COCO file
                rgb_image.shape[:2]
            )
            
            return {
                'rgb': rgb_image,
                'depth': upscaled_depth,
                'plate_mask': plate_mask,
                'frame_id': frame_id
            }
            
        except Exception as e:
            logger.error(f"Error loading data for frame {frame_id}: {str(e)}")
            raise
            
    def process_single_image(self, frame_id: str) -> Dict:
        """Process a single image through the pipeline"""
        try:
            logger.info(f"Processing frame {frame_id}")
            
            # Step 1: Load data
            data = self.load_data(frame_id)
            logger.info("Data loaded successfully")
            
            # Step 2: Calculate camera intrinsics using plate
            intrinsic_params = self.calibrator.calculate_intrinsics(data['plate_mask'])
            logger.info("Camera calibration completed")
            
            # Step 3: Clean depth data
            cleaned_depth = self.noise_reducer.process_depth(
                data['depth'],
                data['plate_mask']
            )
            
            # Step 4: Get depth scale factor using plate
            plate_depth = cleaned_depth[data['plate_mask'] > 0]
            depth_scale = self.calibrator.get_depth_scale_factor(plate_depth)
            cleaned_depth *= depth_scale
            logger.info(f"Depth scaling applied (scale factor: {depth_scale:.4f})")
            
            # Step 5: Process each object
            annotations = self.coco_handler.get_image_annotations(frame_id)
            processed_objects = {}
            
            for ann in annotations:
                category_id = ann['category_id']
                category_name = self.coco_handler.categories[category_id]
                
                # Create object mask
                obj_mask = self.coco_handler.create_mask(
                    ann, 
                    cleaned_depth.shape
                )
                
                # Extract and clean object depth
                obj_depth = cleaned_depth.copy()
                obj_depth[obj_mask == 0] = 0
                
                processed_objects[category_name] = {
                    'mask': obj_mask,
                    'depth': obj_depth,
                    'category_id': category_id,
                    'bbox': ann['bbox']
                }
                
            logger.info(f"Processed {len(processed_objects)} objects")
            
            # Prepare results
            results = {
                'frame_id': frame_id,
                'intrinsic_params': intrinsic_params,
                'depth': cleaned_depth,
                'depth_scale': depth_scale,
                'processed_objects': processed_objects,
                'rgb': data['rgb']
            }
            
            # Save results
            self.save_results(results)
            logger.info(f"Processing completed for frame {frame_id}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_id}: {str(e)}")
            raise
            
    def save_results(self, results: Dict) -> None:
        """Save processed results to upscaled directory"""
        frame_id = results['frame_id']
        
        # Create upscaled directory if it doesn't exist
        upscaled_dir = self.data_dir / 'upscaled'
        upscaled_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filenames
        base_filename = f"depth_frame_{frame_id}"
        
        # Save depth data
        np.save(
            upscaled_dir / f"{base_filename}_upscaled.npy",
            results['depth']
        )
        
        # Save metadata
        metadata = {
            'intrinsic_params': results['intrinsic_params'],
            'depth_scale': float(results['depth_scale']),
            'processed_objects': {}
        }
        
        # Save object-specific data
        for category, obj_data in results['processed_objects'].items():
            metadata['processed_objects'][category] = {
                'category_id': obj_data['category_id'],
                'bbox': obj_data['bbox']
            }
            
            # Save object depth and mask
            obj_prefix = f"{base_filename}_{category}"
            np.save(
                upscaled_dir / f"{obj_prefix}_depth.npy",
                obj_data['depth']
            )
            np.save(
                upscaled_dir / f"{obj_prefix}_mask.npy",
                obj_data['mask']
            )
        
        # Save metadata
        with open(upscaled_dir / f"{base_filename}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)
            
        logger.info(f"Results saved to {upscaled_dir}")

def run_preprocessing(config_path: str):
    """Run the complete preprocessing pipeline"""
    try:
        # Load configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Validate configuration
        required_keys = [
            'data_dir', 'output_dir', 'coco_file',
            'rgb_shape', 'camera_height', 'plate_diameter', 'plate_height'
        ]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
                
        # Initialize and run pipeline
        pipeline = PreprocessingPipeline(config)
        
        # Process each frame
        for frame_id in config['frame_ids']:
            try:
                pipeline.process_single_image(frame_id)
                logger.info(f"Successfully processed frame {frame_id}")
            except Exception as e:
                logger.error(f"Failed to process frame {frame_id}: {str(e)}")
                continue
                
        logger.info("Preprocessing pipeline completed")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline")
    parser.add_argument('--config', required=True, help='Path to config file')
    args = parser.parse_args()
    
    run_preprocessing(args.config)
