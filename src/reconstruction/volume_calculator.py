import numpy as np
from typing import Dict, Tuple, Optional
import logging
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VolumeCalculator:
    def __init__(self, 
                 camera_height: float = 33.0,
                 plate_diameter: float = 25.5):
        """
        Initialize with camera and reference object parameters.
        Following pinhole camera model equations:
        X = (x-cx)Z/fx
        Y = (y-cy)Z/fy
        Volume = Σ(Z(x,y) - Zplate(x,y)) * dA
        """
        self.camera_height = camera_height
        self.plate_diameter = plate_diameter
        self.plate_height = 0.7
        self.CM3_TO_CUPS = 0.0338140225
    def calculate_volume(self, depth_map: np.ndarray, 
                        mask: np.ndarray,
                        plate_height: float,
                        intrinsic_params: Dict,
                        calibration: Optional[Dict] = None) -> Dict[str, float]:
        try:
            pixel_size = intrinsic_params['pixel_size']
            
            # Get the masked depths
            masked_depths = depth_map[mask > 0]
            
            # Calculate base height (where plate touches table)
            plate_base = plate_height + self.plate_height
            
            # Calculate heights relative to plate base
            heights = plate_base - masked_depths
            
            # Only consider positive heights for volume calculation
            valid_heights = heights[heights > 0]
            
            # Debug print
            logger.info(f"Debug - Object Stats:")
            logger.info(f"Raw depths range: [{masked_depths.min():.2f}, {masked_depths.max():.2f}]")
            logger.info(f"Plate surface height: {plate_height:.2f}")
            logger.info(f"Plate base height: {plate_base:.2f}")
            logger.info(f"Valid heights range: [{valid_heights.min():.2f}, {valid_heights.max():.2f}]")
            logger.info(f"Number of total points: {len(heights)}")
            logger.info(f"Number of valid points: {len(valid_heights)}")
            
            # Calculate base area
            base_area = np.sum(mask) * (pixel_size ** 2)
            
            # Calculate volume using only valid (positive) heights
            volume_cm3 = np.sum(valid_heights) * (pixel_size ** 2)
            
            # Apply calibration if provided
            if calibration and 'scale_factor' in calibration:
                volume_cm3 *= calibration['scale_factor']
                
            # Convert to cups
            volume_cups = volume_cm3 * self.CM3_TO_CUPS
            
            # Calculate statistics using valid heights only
            avg_height = np.mean(valid_heights)
            max_height = np.max(valid_heights)
            
            logger.info(
                f"Volume Calculation Results:\n"
                f"Average Height: {avg_height:.2f} cm\n"
                f"Max Height: {max_height:.2f} cm\n"
                f"Base Area: {base_area:.2f} cm²\n"
                f"Volume: {volume_cm3:.2f} cm³ ({volume_cups:.2f} cups)"
            )
            
            return {
                'volume_cm3': float(volume_cm3),
                'volume_cups': float(volume_cups),
                'uncertainty_cm3': float(volume_cm3 * 0.1),
                'uncertainty_cups': float(volume_cups * 0.1),
                'base_area_cm2': float(base_area),
                'avg_height_cm': float(avg_height),
                'max_height_cm': float(max_height)
            }
            
        except Exception as e:
            logger.error(f"Error calculating volume: {str(e)}")
            raise
    def calculate_plate_reference(self, depth_map: np.ndarray,
                                plate_mask: np.ndarray,
                                intrinsic_params: Dict) -> Dict[str, float]:
        """Calculate reference measurements using plate and projection equations"""
        try:
            # Get plate depth values
            plate_depths = depth_map[plate_mask > 0]
            plate_height = np.median(plate_depths)
            
            # Calculate actual plate volume for reference
            actual_volume = np.pi * (self.plate_diameter/2)**2 * self.plate_height
            
            # Calculate estimated plate volume
            plate_base = plate_height + self.plate_height
            plate_heights = plate_base - plate_depths
            valid_heights = plate_heights[plate_heights > 0]
            pixel_size = intrinsic_params['pixel_size']
            estimated_volume = np.sum(valid_heights) * (pixel_size ** 2)
            
            # Scale factor is ratio of actual to estimated volume
            scale_factor = actual_volume / estimated_volume
            
            logger.info(
                f"Plate Calibration:\n"
                f"Actual Plate Volume: {actual_volume:.2f} cm³\n"
                f"Estimated Plate Volume: {estimated_volume:.2f} cm³\n"
                f"Scale Factor: {scale_factor:.4f}\n"
                f"Reference Height: {plate_height:.2f} cm"
            )
            
            return {
                'scale_factor': float(scale_factor),
                'plate_height': float(plate_height)
            }
            
        except Exception as e:
            logger.error(f"Error in plate calibration: {str(e)}")
            raise
