import numpy as np
import cv2
from typing import Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DepthNoiseReducer:
    """
    Handles noise reduction and cleaning of depth data from RGBD images.
    """
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize with optional configuration parameters.
        """
        self.config = config or {
            'bilateral_d': 5,
            'bilateral_sigma_color': 0.1,
            'bilateral_sigma_space': 5.0,
            'median_kernel': 5,
            'outlier_threshold': 2.0
        }
        
    def remove_outliers(self, depth_data: np.ndarray, 
                       mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Remove outlier depth values using statistical analysis.
        """
        if mask is not None:
            valid_depths = depth_data[mask > 0]
        else:
            valid_depths = depth_data[depth_data > 0]
            
        if len(valid_depths) == 0:
            return depth_data
            
        mean_depth = np.mean(valid_depths)
        std_depth = np.std(valid_depths)
        threshold = std_depth * self.config['outlier_threshold']
        
        outliers = np.abs(depth_data - mean_depth) > threshold
        
        cleaned_depth = depth_data.copy()
        if np.any(outliers):
            kernel_size = self.config['median_kernel']
            local_median = cv2.medianBlur(
                depth_data.astype(np.float32),
                kernel_size
            )
            cleaned_depth[outliers] = local_median[outliers]
            
            logger.info(f"Removed {np.sum(outliers)} outlier points")
            
        return cleaned_depth
        
    def fill_missing_values(self, depth_data: np.ndarray) -> np.ndarray:
        """
        Fill missing or invalid depth values using interpolation.
        """
        invalid_mask = (depth_data <= 0) | np.isnan(depth_data)
        
        if not np.any(invalid_mask):
            return depth_data
            
        filled_depth = depth_data.copy()
        
        filled_depth = cv2.inpaint(
            filled_depth.astype(np.float32),
            invalid_mask.astype(np.uint8),
            3,
            cv2.INPAINT_NS
        )
        
        logger.info(f"Filled {np.sum(invalid_mask)} missing values")
        return filled_depth
        
    def apply_bilateral_filter(self, depth_data: np.ndarray) -> np.ndarray:
        """
        Apply bilateral filtering to reduce noise while preserving edges.
        """
        filtered_depth = cv2.bilateralFilter(
            depth_data.astype(np.float32),
            self.config['bilateral_d'],
            self.config['bilateral_sigma_color'],
            self.config['bilateral_sigma_space']
        )
        
        return filtered_depth
        
    def smooth_edges(self, depth_data: np.ndarray, 
                    mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Smooth depth values at object edges.
        """
        if mask is not None:
            edges = cv2.Canny(mask.astype(np.uint8), 100, 200)
            
            kernel = np.ones((3,3), np.uint8)
            edge_region = cv2.dilate(edges, kernel, iterations=1)
            
            smoothed = cv2.GaussianBlur(
                depth_data.astype(np.float32),
                (5,5),
                1.0
            )
            
            result = depth_data.copy()
            result[edge_region > 0] = smoothed[edge_region > 0]
            
            return result
        
        return depth_data
        
    def process_depth(self, depth_data: np.ndarray,
                     mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply complete noise reduction pipeline to depth data.
        """
        logger.info("Starting depth noise reduction")
        
        filled_depth = self.fill_missing_values(depth_data)
        
        cleaned_depth = self.remove_outliers(filled_depth, mask)
        
        filtered_depth = self.apply_bilateral_filter(cleaned_depth)
        
        if mask is not None:
            final_depth = self.smooth_edges(filtered_depth, mask)
        else:
            final_depth = filtered_depth
            
        logger.info("Completed depth noise reduction")
        return final_depth
