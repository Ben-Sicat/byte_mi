import numpy as np
import cv2
from typing import Tuple, Optional, Dict
import logging
from pathlib import Path
from ..utils.io_utils import load_metadata, get_frame_dimensions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DepthProcessor:
    """
    Handles depth data processing and image alignment.
    All measurements in meters.
    """
    
    def __init__(self, rgbd_meta_path: Path, rgb_meta_path: Path):
        """
        Initialize processor with metadata paths.
        """
        # Load dimensions from metadata
        self.depth_shape, self.rgb_shape = get_frame_dimensions(
            rgbd_meta_path, 
            rgb_meta_path
        )
        
        # Load depth metadata
        self.depth_meta = load_metadata(rgbd_meta_path)
        self.dtype = np.uint16
        
        # Processing parameters
        self.bilateral_d = 5
        self.bilateral_sigma_color = 50
        self.bilateral_sigma_space = 50
        
        logger.info(
            f"Initialized DepthProcessor with shapes - "
            f"Depth: {self.depth_shape}, RGB: {self.rgb_shape}"
        )
        
    def load_raw_depth(self, file_path: str) -> np.ndarray:
        """Load raw depth data."""
        try:
            raw_data = np.fromfile(file_path, dtype=self.dtype)
            
            expected_size = self.depth_shape[0] * self.depth_shape[1]
            if raw_data.size != expected_size:
                raise ValueError(
                    f"Raw data size {raw_data.size} does not match "
                    f"expected size {expected_size}"
                )
            
            depth_data = raw_data.reshape(self.depth_shape)
            
            logger.info(f"Loaded depth data - Shape: {depth_data.shape}, "
                       f"Range: [{depth_data.min()}, {depth_data.max()}]")
            
            return depth_data
            
        except Exception as e:
            logger.error(f"Error loading depth file: {str(e)}")
            raise
    def align_to_depth(self, rgb_image: Optional[np.ndarray] = None, 
            mask: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Align RGB image and/or mask to depth resolution.
        
        Args:
            rgb_image: Optional RGB image at original resolution
            mask: Optional binary mask at original resolution
            
        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: 
                RGB and mask at depth resolution (None for any not provided)
        """
        # Resize RGB image if provided
        aligned_rgb = None
        if rgb_image is not None:
            aligned_rgb = cv2.resize(
                rgb_image,
                (self.depth_shape[1], self.depth_shape[0]),  # width, height
                interpolation=cv2.INTER_AREA  # Better for downscaling
            )
        
        # Resize mask if provided
        aligned_mask = None
        if mask is not None:
            aligned_mask = cv2.resize(
                mask.astype(np.uint8),
                (self.depth_shape[1], self.depth_shape[0]),
                interpolation=cv2.INTER_NEAREST  # Preserve binary values
            ).astype(bool)  # Convert back to boolean
            
            # Log the mask alignment results
            if np.any(mask) and np.any(aligned_mask):
                original_pixels = np.sum(mask)
                aligned_pixels = np.sum(aligned_mask)
                logger.info(
                    f"Aligned mask - Original: {original_pixels} pixels, "
                    f"Aligned: {aligned_pixels} pixels"
                )
        
        return aligned_rgb, aligned_mask   
    def process_depth(self, depth_data: np.ndarray) -> np.ndarray:
        """Process depth data to remove noise."""
        if depth_data.shape != self.depth_shape:
            raise ValueError(f"Expected shape {self.depth_shape}, got {depth_data.shape}")
            
        # Convert to float32 for processing
        depth = depth_data.astype(np.float32)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered_depth = cv2.bilateralFilter(
            depth,
            d=self.bilateral_d,
            sigmaColor=self.bilateral_sigma_color,
            sigmaSpace=self.bilateral_sigma_space
        )
        
        return filtered_depth
