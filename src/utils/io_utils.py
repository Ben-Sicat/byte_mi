import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

def load_metadata(file_path: Path) -> Dict:
    """Load metadata from a .meta file."""
    try:
        with open(file_path, 'r') as f:
            metadata = json.load(f)
            
        required_keys = ['width', 'height']
        if not all(key in metadata for key in required_keys):
            raise ValueError(f"Missing required keys in metadata: {required_keys}")
            
        return metadata
    except Exception as e:
        logger.error(f"Error loading metadata from {file_path}: {str(e)}")
        raise

def get_frame_dimensions(rgbd_meta_path: Path, rgb_meta_path: Path) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Get frame dimensions from metadata files."""
    try:
        rgbd_meta = load_metadata(rgbd_meta_path)
        rgb_meta = load_metadata(rgb_meta_path)
        
        depth_dims = (rgbd_meta['height'], rgbd_meta['width'])
        rgb_dims = (rgb_meta['height'], rgb_meta['width'])
        
        # Validate dimensions
        if not all(d > 0 for d in depth_dims + rgb_dims):
            raise ValueError("Invalid dimensions: all dimensions must be positive")
            
        logger.info(f"Loaded dimensions - Depth: {depth_dims}, RGB: {rgb_dims}")
        return depth_dims, rgb_dims
    except Exception as e:
        logger.error(f"Error getting frame dimensions: {str(e)}")
        raise

def validate_image_alignment(source: np.ndarray, aligned: np.ndarray, 
                           target_shape: Tuple[int, int], 
                           threshold: float = 0.1) -> bool:
    """
    Validate image alignment by checking dimensions and content.
    
    Args:
        source: Original image
        aligned: Aligned image
        target_shape: Expected shape after alignment
        threshold: Maximum allowed mean absolute difference after normalization
        
    Returns:
        bool: True if alignment is valid
    """
    try:
        # Check dimensions
        if aligned.shape[:2] != target_shape:
            logger.error(f"Invalid aligned shape: {aligned.shape[:2]} != {target_shape}")
            return False
            
        # For masks, check binary values are preserved
        if aligned.dtype == bool or (aligned.dtype == np.uint8 and np.max(aligned) == 1):
            if not np.array_equal(np.unique(aligned), np.unique(source)):
                logger.error("Binary mask values were not preserved during alignment")
                return False
                
        # For RGB images, check content preservation
        else:
            # Resize source to target for comparison
            source_resized = cv2.resize(source, target_shape[::-1])
            
            # Normalize and compare
            source_norm = source_resized.astype(float) / np.max(source_resized)
            aligned_norm = aligned.astype(float) / np.max(aligned)
            
            diff = np.mean(np.abs(source_norm - aligned_norm))
            if diff > threshold:
                logger.error(f"Alignment error too high: {diff:.3f} > {threshold}")
                return False
                
        return True
        
    except Exception as e:
        logger.error(f"Error validating alignment: {str(e)}")
        return False
def validate_depth_data(depth_data: np.ndarray, 
                       expected_shape: Tuple[int, int],
                       metadata: Optional[Dict] = None) -> bool:
    """Validate depth data against expected parameters."""
    try:
        # Check dimensions
        if depth_data.shape != expected_shape:
            logger.error(f"Invalid depth shape: {depth_data.shape} != {expected_shape}")
            return False
            
        # Check data type
        if depth_data.dtype not in [np.uint16, np.float32]:
            logger.error(f"Invalid depth dtype: {depth_data.dtype}")
            return False
            
        # Check value range
        if np.all(depth_data == 0):
            logger.error("Depth data is all zeros")
            return False
            
        # If metadata provided, check against expected ranges with tolerance
        if metadata:
            min_depth = metadata.get('minDepth')
            max_depth = metadata.get('maxDepth')
            
            if min_depth is not None and max_depth is not None:
                # Allow for some tolerance in the depth range
                tolerance = 0.5  # 50% tolerance
                min_allowed = min_depth * (1 - tolerance)
                max_allowed = max_depth * (1 + tolerance)
                
                actual_min = np.min(depth_data[depth_data > 0])
                actual_max = np.max(depth_data)
                
                if actual_min < min_allowed or actual_max > max_allowed:
                    logger.warning(
                        f"Depth values outside expected range: "
                        f"[{actual_min:.3f}, {actual_max:.3f}] vs "
                        f"[{min_depth:.3f}, {max_depth:.3f}]"
                    )
                    # Don't fail validation for range issues, just warn
                    
        return True
        
    except Exception as e:
        logger.error(f"Error validating depth data: {str(e)}")
        return False
