import numpy as np
import cv2
from typing import Tuple, Optional, Dict
import logging
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DepthProcessor:
    """
    Handles loading and processing of raw depth data.
    All measurements in centimeters.
    """
    
    def __init__(self):
        # Raw depth data properties
        self.depth_shape = (120, 120)  # Original depth resolution
        self.target_shape = (480, 640)  # RGB resolution
        self.dtype = np.uint16        # Raw data type
        
        # Processing parameters
        self.bilateral_d = 5
        self.bilateral_sigma_color = 50
        self.bilateral_sigma_space = 50
        
    def load_raw_rgbd(self, file_path: str) -> np.ndarray:
        """
        Load raw depth data from .raw file.
        
        Args:
            file_path: Path to .raw depth file
            
        Returns:
            np.ndarray: Depth data array (120, 120)
        """
        try:
            # Load raw data
            raw_data = np.fromfile(file_path, dtype=self.dtype)
            
            expected_size = self.depth_shape[0] * self.depth_shape[1]
            if raw_data.size != expected_size:
                raise ValueError(
                    f"Raw data size {raw_data.size} does not match "
                    f"expected size {expected_size}"
                )
            
            # Reshape to 2D array
            depth_data = raw_data.reshape(self.depth_shape)
            
            logger.info(f"Loaded depth data - Shape: {depth_data.shape}, "
                       f"Range: [{depth_data.min()}, {depth_data.max()}]")
            
            return depth_data
            
        except Exception as e:
            logger.error(f"Error loading depth file: {str(e)}")
            raise
            
    def process_depth(self, depth_data: np.ndarray) -> np.ndarray:
        """
        Process depth data to remove noise and fill holes.
        
        Args:
            depth_data: Raw depth data
            
        Returns:
            np.ndarray: Processed depth data
        """
        if depth_data.shape != self.depth_shape:
            raise ValueError(f"Expected shape {self.depth_shape}, got {depth_data.shape}")
            
        # Convert to float32 for processing
        depth = depth_data.astype(np.float32)
        
        # Remove outliers
        valid_mask = depth > 0
        if np.any(valid_mask):
            mean_depth = np.mean(depth[valid_mask])
            std_depth = np.std(depth[valid_mask])
            
            # Define valid range
            min_valid = mean_depth - 2 * std_depth
            max_valid = mean_depth + 2 * std_depth
            
            # Create outlier mask
            outlier_mask = (depth < min_valid) | (depth > max_valid)
            depth[outlier_mask] = 0
            
            logger.info(f"Removed {np.sum(outlier_mask)} outlier points")
        
        # Fill holes using nearest neighbor interpolation
        zero_mask = depth == 0
        if np.any(zero_mask):
            valid_points = np.argwhere(~zero_mask)
            invalid_points = np.argwhere(zero_mask)
            
            if len(valid_points) > 0:
                tree = cKDTree(valid_points)
                _, indices = tree.query(invalid_points)
                
                depth[invalid_points[:, 0], invalid_points[:, 1]] = \
                    depth[valid_points[indices][:, 0], valid_points[indices][:, 1]]
                    
                logger.info(f"Filled {len(invalid_points)} holes")
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered_depth = cv2.bilateralFilter(
            depth,
            d=self.bilateral_d,
            sigmaColor=self.bilateral_sigma_color,
            sigmaSpace=self.bilateral_sigma_space
        )
        
        return filtered_depth
        
    def upscale_depth(self, depth_data: np.ndarray) -> np.ndarray:
        """
        Upscale depth data to match RGB resolution.
        
        Args:
            depth_data: Processed depth data (120, 120)
            
        Returns:
            np.ndarray: Upscaled depth data (480, 640)
        """
        if depth_data.shape != self.depth_shape:
            raise ValueError(f"Expected shape {self.depth_shape}, got {depth_data.shape}")
        
        # First upscale to square maintaining aspect ratio
        square_size = self.target_shape[0]  # 480
        interim = cv2.resize(
            depth_data,
            (square_size, square_size),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Create target array
        upscaled = np.zeros(self.target_shape, dtype=depth_data.dtype)
        
        # Calculate padding for centering
        pad_left = (self.target_shape[1] - square_size) // 2  # (640 - 480) // 2
        
        # Place upscaled data in center
        upscaled[:, pad_left:pad_left+square_size] = interim
        
        logger.info(f"Upscaled depth data from {depth_data.shape} to {upscaled.shape}")
        
        return upscaled
        
    def validate_depth(self, depth_data: np.ndarray) -> bool:
        """
        Validate depth data.
        
        Args:
            depth_data: Depth data to validate
            
        Returns:
            bool: True if depth data is valid
        """
        try:
            if depth_data.ndim != 2:
                logger.error(f"Invalid dimensions: {depth_data.ndim}")
                return False
                
            if depth_data.size == 0:
                logger.error("Empty depth data")
                return False
                
            if not np.any(depth_data > 0):
                logger.error("No valid depth values")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating depth data: {str(e)}")
            return False
            
    def get_depth_stats(self, depth_data: np.ndarray) -> Dict:
        """
        Get statistics about depth data.
        
        Args:
            depth_data: Depth data
            
        Returns:
            Dict: Statistics about depth data
        """
        valid_mask = depth_data > 0
        if not np.any(valid_mask):
            return {
                'min': 0,
                'max': 0,
                'mean': 0,
                'std': 0,
                'valid_points': 0
            }
            
        valid_depths = depth_data[valid_mask]
        return {
            'min': float(np.min(valid_depths)),
            'max': float(np.max(valid_depths)),
            'mean': float(np.mean(valid_depths)),
            'std': float(np.std(valid_depths)),
            'valid_points': int(np.sum(valid_mask))
        }
