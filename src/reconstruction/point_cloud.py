import numpy as np
from typing import Dict, Tuple, Optional
import logging
from scipy.spatial import ConvexHull

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PointCloud:
    """
    Handles conversion of depth maps to 3D point clouds and basic measurements.
    Uses pinhole camera model for 3D reconstruction.
    """
    
    def __init__(self, intrinsic_params: Dict):
        """
        Initialize with camera intrinsic parameters.
        
        Args:
            intrinsic_params: Dictionary containing:
                - focal_length: in pixels
                - pixel_size: in cm/pixel
                - principal_point: (x,y) in pixels
        """
        self.focal_length = intrinsic_params['focal_length']
        self.pixel_size = intrinsic_params['pixel_size']
        self.principal_point = intrinsic_params['principal_point']
        
        logger.info(
            f"Initialized PointCloud with focal length: {self.focal_length:.2f}, "
            f"pixel size: {self.pixel_size:.4f}"
        )
    def depth_to_point_cloud(self, depth_map: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Convert depth map to 3D point cloud using pinhole model."""
        # Get pixel coordinates
        rows, cols = depth_map.shape
        y_indices, x_indices = np.meshgrid(
            np.arange(rows), 
            np.arange(cols), 
            indexing='ij'
        )
        
        x_indices = x_indices.flatten()
        y_indices = y_indices.flatten()
        depth_values = depth_map.flatten()
        
        if mask is not None:
            mask_flat = mask.flatten()
            valid_points = mask_flat > 0
            x_indices = x_indices[valid_points]
            y_indices = y_indices[valid_points]
            depth_values = depth_values[valid_points]
        
        x_centered = (x_indices - self.principal_point[0]) * self.pixel_size
        y_centered = (y_indices - self.principal_point[1]) * self.pixel_size
        
        X = x_centered * depth_values / self.focal_length
        Y = y_centered * depth_values / self.focal_length
        Z = depth_values
        
        points = np.column_stack([X, Y, Z])
        
        logger.info(f"Generated point cloud with {len(points)} points")
        return points   
    def estimate_volume(self, points: np.ndarray, method: str = 'convex_hull') -> float:
        """
        Estimate volume of point cloud.
        """
        if len(points) < 4:
            raise ValueError("Need at least 4 points to estimate volume")
            
        if method == 'convex_hull':
            hull = ConvexHull(points)
            return hull.volume
        elif method == 'alpha_shape':
            raise NotImplementedError("Alpha shape method not implemented")
        else:
            raise ValueError(f"Unknown volume estimation method: {method}")
            
    def calculate_surface_area(self, points: np.ndarray) -> float:
        """
        Calculate surface area of point cloud using convex hull.
        """
        if len(points) < 4:
            raise ValueError("Need at least 4 points to calculate surface area")
            
        hull = ConvexHull(points)
        return hull.area
        
    def get_dimensions(self, points: np.ndarray) -> Dict[str, float]:
        """
        Calculate bounding box dimensions.
        """
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        dimensions = max_coords - min_coords
        
        return {
            'length': float(dimensions[0]),
            'width': float(dimensions[1]),
            'height': float(dimensions[2])
        }
