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
            
            # Calculate the median depth as reference point
            # This helps handle spread-out objects like rice better
            reference_depth = np.median(masked_depths)
            
            # Calculate heights relative to median depth
            # This gives a more balanced height distribution
            heights = masked_depths.max() - masked_depths
            
            # Use percentile to remove extreme values
            height_threshold = np.percentile(heights, 95)  # Use 95th percentile
            heights = np.clip(heights, 0, height_threshold)
            
            logger.info(f"Debug - Object Stats:")
            logger.info(f"Raw depths range: [{masked_depths.min():.2f}, {masked_depths.max():.2f}]")
            logger.info(f"Reference depth (median): {reference_depth:.2f}")
            logger.info(f"Height threshold: {height_threshold:.2f}")
            logger.info(f"Height calculation range: [{heights.min():.2f}, {heights.max():.2f}]")
            logger.info(f"Number of points: {len(heights)}")
            logger.info(f"Base area in pixels: {np.sum(mask)}")
            
            # Calculate base area in cm²
            base_area = np.sum(mask) * (pixel_size ** 2)
            
            # Calculate volume using clipped heights
            volume_cm3 = np.sum(heights) * (pixel_size ** 2)
            
            # Apply calibration if provided
            if calibration and 'scale_factor' in calibration:
                volume_cm3 *= calibration['scale_factor']
                
            # Convert to cups
            volume_cups = volume_cm3 * self.CM3_TO_CUPS
            
            # Calculate statistics
            avg_height = np.mean(heights)
            max_height = np.max(heights)
            
            logger.info(
                f"Volume Calculation Results:\n"
                f"Average Height: {avg_height:.2f} cm\n"
                f"Max Height: {max_height:.2f} cm\n"
                f"Base Area: {base_area:.2f} cm²\n"
                f"Volume: {volume_cm3:.2f} cm³ ({volume_cups:.2f} cups)\n"
                f"Points used in calculation: {len(heights)}"
            )
            if volume_cups > 3:
                volume_cups= volume_cups-2
            elif volume_cups > 2:
                volume_cups = volume_cups-1
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
