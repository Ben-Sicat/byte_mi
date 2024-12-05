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
        """
        Calculate volume using pinhole camera model equations.
        Volume = Σ(Z(x,y) - Zplate(x,y)) * dA
        where dA is the area of each pixel in world coordinates
        """
        try:
            # Get camera parameters
            fx = intrinsic_params['focal_length']
            pixel_size = intrinsic_params['pixel_size']
            cx, cy = intrinsic_params['principal_point']
            
            # Get pixel coordinates
            y_indices, x_indices = np.nonzero(mask)
            
            # Get depth values for valid pixels
            z_values = depth_map[y_indices, x_indices]
            z_plate = plate_height
            
            # Calculate differential area (dA) for each pixel
            # dA = (X*Y)/Z² * pixel_size²
            x_world = (x_indices - cx) * z_values / fx
            y_world = (y_indices - cy) * z_values / fx  # Assuming fx = fy
            
            # Calculate area of each pixel in world coordinates
            dA = pixel_size * pixel_size
            
            # Calculate volume using the formula
            heights = z_values - z_plate
            heights[heights < 0] = 0
            
            volume_cm3 = np.sum(heights * dA)
            
            # Apply calibration if provided
            if calibration and 'scale_factor' in calibration:
                volume_cm3 *= calibration['scale_factor']
            
            # Convert to cups
            volume_cups = volume_cm3 * self.CM3_TO_CUPS
            
            # Calculate statistics
            avg_height = np.mean(heights[heights > 0])
            max_height = np.max(heights)
            
            logger.info(
                f"Volume Calculation Results:\n"
                f"Average Height Above Plate: {avg_height:.2f} cm\n"
                f"Max Height Above Plate: {max_height:.2f} cm\n"
                f"Volume: {volume_cm3:.2f} cm³ ({volume_cups:.2f} cups)"
            )
            
            return {
                'volume_cm3': float(volume_cm3),
                'volume_cups': float(volume_cups),
                'uncertainty_cm3': float(volume_cm3 * 0.1),
                'uncertainty_cups': float(volume_cups * 0.1),
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
            plate_depths = depth_map[plate_mask > 0]
            plate_height = np.median(plate_depths)
            
            plate_pixel = plate_height  # depth value from depth map
            plate_real = self.camera_height  # actual height of camera
            scale_factor = (plate_real / plate_pixel) * 0.33  # Added adjustment factor
            
            logger.info(
                f"Plate Calibration:\n"
                f"Camera Height (plate_real): {plate_real:.2f} cm\n"
                f"Depth Value (plate_pixel): {plate_pixel:.2f}\n"
                f"Scale Factor (plate_real/plate_pixel * 0.33): {scale_factor:.4f}\n"
                f"Reference Height: {plate_height:.2f} cm"
            )
            
            return {
                'scale_factor': float(scale_factor),
                'plate_height': float(plate_height),
                'camera_height': float(plate_real),
                'depth_value': float(plate_pixel)
            }
            
        except Exception as e:
            logger.error(f"Error in plate calibration: {str(e)}")
            raise           
