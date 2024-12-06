import sys
from pathlib import Path
import argparse
import logging
import json
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.preprocessing.preprocessing import PreprocessingPipeline
from src.reconstruction.volume_calculator import VolumeCalculator
from src.utils.logging_utils import setup_logging

def main():
    parser = argparse.ArgumentParser(description="Run volume estimation pipeline")
    parser.add_argument('--config', required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(log_dir="logs", log_prefix="volume_estimation")
    
    try:
        # Load config
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        # 1. Run preprocessing
        logger.info("Starting preprocessing pipeline...")
        pipeline = PreprocessingPipeline(config)
        
        for frame_id in config['frame_ids']:
            # Process single image
            result = pipeline.process_single_image(frame_id)
            
            # 2. Calculate volumes
            logger.info("\nCalculating volumes...")
            calc = VolumeCalculator(
                camera_height=config.get('camera_height', 33.0),
                plate_diameter=config.get('plate_diameter', 25.5)
            )
            
            # Get plate data
            plate_data = next(
                (data for name, data in result['processed_objects'].items() 
                 if name == 'plate'),
                None
            )
            
            if plate_data is None:
                raise ValueError("No plate found in processed objects")
            
            # Get calibration and measured plate height
            calibration = calc.calculate_plate_reference(
                depth_map=result['depth'],
                plate_mask=plate_data['mask'],
                intrinsic_params=result['intrinsic_params']
            )
            
            plate_height = calibration['plate_height']
            
            # Calculate volumes for each food item
            volume_results = {}
            for obj_name, obj_data in result['processed_objects'].items():
                if obj_name == 'plate':
                    continue
                    
                logger.info(f"\nProcessing {obj_name}...")
                volume_data = calc.calculate_volume(
                    depth_map=result['depth'],
                    mask=obj_data['mask'],
                    plate_height=plate_height,  # Use measured plate height
                    intrinsic_params=result['intrinsic_params'],
                    calibration=calibration
                )
                
                volume_results[obj_name] = volume_data
            
            # Save results
            output_dir = Path(config['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / f"volumes_{frame_id}.json"
            with open(output_file, 'w') as f:
                json.dump(volume_results, f, indent=2)
                
            logger.info(f"\nResults saved to {output_file}")
            logger.info("\nVolume Summary:")
            for obj_name, data in volume_results.items():
                logger.info(
                    f"{obj_name}: {data['volume_cups']:.2f} cups "
                    f"(±{data['uncertainty_cups']:.2f})"
                )
                
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    main()
