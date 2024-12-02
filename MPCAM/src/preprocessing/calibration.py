import numpy as np
import cv2
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CameraCalibrator:
    """
    Calculate intrinsic parameters using known measurements and plate as reference.
    All measurements are in centimeters.
    """
    def __init__(self):
        # Known measurements in cm
        self.camera_height = 33.0  
        self.plate_diameter = 25.5  
        self.plate_height = 0.7  
        
        # Will be calculated
        self.focal_length = None
        self.principal_point = None
        self.pixel_size = None
        
    def calculate_focal_length(self, plate_diameter_pixels: float) -> float:
        """
        Calculate focal length using pinhole model and plate as reference.
        f = (P * H) / W
        where:
        f = focal length in pixels
        P = plate diameter in pixels
        H = camera height in cm
        W = actual plate diameter in cm
        """
        focal_length = (plate_diameter_pixels * self.camera_height) / self.plate_diameter
        logger.info(f"Calculated focal length: {focal_length:.2f} pixels")
        return focal_length
        
    def calculate_pixel_size(self, plate_diameter_pixels: float) -> float:
        """
        Calculate pixel size in cm
        pixel_size = actual_size / pixel_size
        """
        pixel_size = self.plate_diameter / plate_diameter_pixels
        logger.info(f"Calculated pixel size: {pixel_size:.6f} cm/pixel")
        return pixel_size

    def get_plate_measurements(self, plate_mask: np.ndarray) -> Dict:
        """
        Get plate measurements from mask in pixels.
        """
        # Find contours of the plate
        contours, _ = cv2.findContours(
            plate_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            raise ValueError("No plate contour found in mask")
        
        # Get largest contour (plate)
        plate_contour = max(contours, key=cv2.contourArea)
        
        # Find the minimum enclosing circle
        (center_x, center_y), radius = cv2.minEnclosingCircle(plate_contour)
        diameter_pixels = radius * 2
        
        return {
            'center': (center_x, center_y),
            'radius': radius,
            'diameter_pixels': diameter_pixels
        }

    def calculate_intrinsics(self, plate_mask: np.ndarray) -> Dict:
        """
        Calculate all intrinsic parameters using plate mask.
        """
        try:
            # get plate measurements
            plate_info = self.get_plate_measurements(plate_mask)
            
            # calculate focal length
            self.focal_length = self.calculate_focal_length(
                plate_info['diameter_pixels']
            )
            
            # calculate pixel size
            self.pixel_size = self.calculate_pixel_size(
                plate_info['diameter_pixels']
            )
            
            # principal point 
            height, width = plate_mask.shape
            self.principal_point = (width / 2, height / 2)
            
            intrinsic_params = {
                'focal_length': self.focal_length,  # in pixels
                'pixel_size': self.pixel_size,      # cm/pixel
                'principal_point': self.principal_point,
                'image_dimensions': (height, width),
                'camera_height': self.camera_height,
                'reference_object': {
                    'type': 'plate',
                    'diameter': self.plate_diameter,
                    'height': self.plate_height,
                    'measured_diameter_pixels': plate_info['diameter_pixels'],
                    'center_pixels': plate_info['center']
                }
            }
            
            self._validate_parameters(intrinsic_params)
            
            return intrinsic_params
            
        except Exception as e:
            logger.error(f"Error calculating intrinsic parameters: {str(e)}")
            raise

    def _validate_parameters(self, params: Dict) -> None:
        """
        Validate calculated parameters.
        """
        if params['focal_length'] <= 0:
            raise ValueError(f"Invalid focal length: {params['focal_length']}")
            
        if params['pixel_size'] <= 0 or params['pixel_size'] > 1:
            raise ValueError(f"Invalid pixel size: {params['pixel_size']}")
            
        measured_diameter_cm = (
            params['reference_object']['measured_diameter_pixels'] * 
            params['pixel_size']
        )
        error_margin = abs(measured_diameter_cm - self.plate_diameter)
        if error_margin > 1.7:  # More than 1cm error
            logger.warning(
                f"Large error in plate diameter measurement: "
                f"{error_margin:.2f}cm"
            )

    def get_depth_scale_factor(self, plate_depth_values: np.ndarray) -> float:
        """
        Calculate depth scale factor using plate as reference.
        """
        # Expected plate distance from camera
        expected_plate_distance = self.camera_height - self.plate_height
        
        # Use median of plate depth values
        measured_plate_distance = np.median(plate_depth_values)
        
        # Calculate scale factor
        scale_factor = expected_plate_distance / measured_plate_distance
        
        logger.info(f"Depth scale factor: {scale_factor:.4f}")
        return scale_factor
