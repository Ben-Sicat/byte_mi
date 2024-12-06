import cv2
import numpy as np
from typing import Dict, Optional
import logging
from pathlib import Path
import json

from ..core.depth_processor import DepthProcessor
from ..utils.io_utils import load_metadata, validate_depth_data, validate_image_alignment
from ..utils.coco_utils import CocoHandler
from ..core.depth_processor import DepthProcessor
from .calibration import CameraCalibrator
from .noise_reduction import DepthNoiseReducer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PreprocessingPipeline:
    def __init__(self, config: Dict):
        """
        Initialize preprocessing pipeline.
        
            config: Dict containing:
                - data_dir: Path to data directory
                - output_dir: Path to save processed data
                - coco_file: Path to COCO annotations
                - camera_height: Height of camera in cm
                - plate_diameter: Diameter of plate in cm
                - plate_height: Height of plate in cm
        """
        self.config = config
        self.data_dir = Path(config['data_dir'])
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.coco_handler = CocoHandler(config['coco_file'])
        self.calibrator = CameraCalibrator()
        self.noise_reducer = DepthNoiseReducer()
        
        logger.info("Initialized preprocessing pipeline")
        
    def load_data(self, frame_id: str) -> Dict:
        """Load all necessary data for processing"""
        try:
            rgbd_meta_path = self.data_dir / "rgbd" / f"depth_frame_{frame_id}.meta"
            rgb_meta_path = self.data_dir / "segmented" / f"rgb_frame_{frame_id}.meta"
            
            self.depth_processor = DepthProcessor(rgbd_meta_path, rgb_meta_path)
            
            rgb_path = self.data_dir / "segmented" / f"rgb_frame_{frame_id}.png"
            if not rgb_path.exists():
                raise FileNotFoundError(f"RGB image not found: {rgb_path}")
                
            rgb_image = cv2.imread(str(rgb_path))
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            
            depth_path = self.data_dir / "rgbd" / f"depth_frame_{frame_id}.raw"
            if not depth_path.exists():
                raise FileNotFoundError(f"Depth data not found: {depth_path}")
                
            depth_meta = load_metadata(rgbd_meta_path)
            
            raw_depth = self.depth_processor.load_raw_depth(str(depth_path))
            
            if not validate_depth_data(raw_depth, self.depth_processor.depth_shape, depth_meta):
                raise ValueError("Invalid depth data")
                
            processed_depth = self.depth_processor.process_depth(raw_depth)
            
            plate_mask = self.coco_handler.create_category_mask(
                frame_id, 
                'plate',
                rgb_image.shape[:2]
            )
            
            aligned_rgb, aligned_mask = self.depth_processor.align_to_depth(
                rgb_image, plate_mask
            )
            
            if not validate_image_alignment(rgb_image, aligned_rgb, self.depth_processor.depth_shape):
                raise ValueError("RGB alignment failed validation")
                
            if not validate_image_alignment(plate_mask, aligned_mask, self.depth_processor.depth_shape):
                raise ValueError("Mask alignment failed validation")
                
            return {
                'rgb': aligned_rgb,
                'depth': processed_depth,
                'plate_mask': aligned_mask,
                'frame_id': frame_id,
                'original_rgb': rgb_image,  # Keep original for reference
                'original_mask': plate_mask  # Keep original for reference
            }
            
        except Exception as e:
            logger.error(f"Error loading data for frame {frame_id}: {str(e)}")
            raise
    
    def process_single_image(self, frame_id: str) -> Dict:
        """Process a single image through the pipeline"""
        try:
            logger.info(f"Processing frame {frame_id}")
            
            data = self.load_data(frame_id)
            logger.info("Data loaded successfully")
            
            intrinsic_params = self.calibrator.calculate_intrinsics(data['plate_mask'])
            logger.info("Camera calibration completed")
            
            cleaned_depth = self.noise_reducer.process_depth(
                data['depth'],
                data['plate_mask']
            )
            
            plate_depth = cleaned_depth[data['plate_mask'] > 0]
            if len(plate_depth) == 0:
                raise ValueError("No valid depth values found in plate region")
                
            depth_scale = self.calibrator.get_depth_scale_factor(plate_depth)
            cleaned_depth *= depth_scale
            logger.info(f"Depth scaling applied (scale factor: {depth_scale:.4f})")
            
            annotations = self.coco_handler.get_image_annotations(frame_id)
            processed_objects = {}
            
            for ann in annotations:
                category_id = ann['category_id']
                category_name = self.coco_handler.categories[category_id]
                
                original_mask = self.coco_handler.create_mask(
                    ann, 
                    (self.depth_processor.rgb_shape[0], self.depth_processor.rgb_shape[1])
                )
                
                if np.any(original_mask):
                    _, aligned_mask = self.depth_processor.align_to_depth(mask=original_mask)
                    
                    if aligned_mask is not None and np.any(aligned_mask):
                        obj_depth = cleaned_depth.copy()
                        obj_depth[~aligned_mask] = 0
                        
                        processed_objects[category_name] = {
                            'mask': aligned_mask,
                            'depth': obj_depth,
                            'category_id': category_id,
                            'bbox': ann['bbox']
                        }
                        
            logger.info(f"Processed {len(processed_objects)} objects")
            
            results = {
                'frame_id': frame_id,
                'intrinsic_params': intrinsic_params,
                'depth': cleaned_depth,
                'depth_scale': depth_scale,
                'processed_objects': processed_objects,
                'rgb': data['rgb']
            }
            
            self.save_results(results)
            logger.info(f"Processing completed for frame {frame_id}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_id}: {str(e)}")
            raise
    def save_results(self, results: Dict) -> None:
        """Save processed results to output directory"""
        frame_id = results['frame_id']
        base_filename = f"depth_frame_{frame_id}"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(
            self.output_dir / f"{base_filename}_processed.npy",
            results['depth']
        )
        
        cv2.imwrite(
            str(self.output_dir / f"{base_filename}_aligned_rgb.png"),
            cv2.cvtColor(results['rgb'], cv2.COLOR_RGB2BGR)
        )
        
        for category, obj_data in results['processed_objects'].items():
            mask_filename = f"{base_filename}_{category}_mask.npy"
            np.save(self.output_dir / mask_filename, obj_data['mask'])
        
        metadata = {
            'intrinsic_params': results['intrinsic_params'],
            'depth_scale': float(results['depth_scale']),
            'processed_objects': {
                category: {
                    'category_id': obj_data['category_id'],
                    'bbox': obj_data['bbox']
                }
                for category, obj_data in results['processed_objects'].items()
            },
            'alignment_info': {
                'depth_shape': self.depth_processor.depth_shape,
                'rgb_shape': self.depth_processor.rgb_shape  # Using original RGB shape from metadata
            }
        }
        
        with open(self.output_dir / f"{base_filename}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)
            
        logger.info(
            f"Saved processed results to {self.output_dir}:\n"
            f"- Processed depth map\n"
            f"- Aligned RGB image\n"
            f"- Object masks: {list(results['processed_objects'].keys())}\n"
            f"- Metadata with alignment info"
        )
def run_preprocessing(config_path: str):
    """Run the complete preprocessing pipeline"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        required_keys = [
            'data_dir', 'output_dir', 'coco_file',
            'rgb_shape', 'camera_height', 'plate_diameter', 'plate_height'
        ]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
                
        pipeline = PreprocessingPipeline(config)
        
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
