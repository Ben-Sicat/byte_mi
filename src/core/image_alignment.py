import numpy as np
import cv2
from typing import Tuple, Dict
import logging
from ..utils.coco_utils import CocoHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageAligner:
    """Handles alignment between RGB images, RGBD data, and segmentation masks"""
    def __init__(self, coco_file: str):
        self.rgb_shape = None
        self.rgbd_shape = None
        self.coco_handler = CocoHandler(coco_file)
        
    def set_reference_sizes(self, rgb_shape: Tuple[int, int], 
                           rgbd_shape: Tuple[int, int]) -> None:
        self.rgb_shape = rgb_shape
        self.rgbd_shape = rgbd_shape
        logger.info(f"Set reference shapes - RGB: {rgb_shape}, RGBD: {rgbd_shape}")
        
    def align_rgbd_to_rgb(self, rgbd_data: np.ndarray) -> np.ndarray:
        """Align RGBD data to RGB dimensions"""
        if not (self.rgb_shape and self.rgbd_shape):
            raise ValueError("Reference sizes not set")
            
        if rgbd_data.shape[2] != 4:
            raise ValueError(f"Expected 4 channels in RGBD data, got {rgbd_data.shape[2]}")
            
        # Split and resize channels
        rgb_channels = rgbd_data[:, :, :3]
        depth_channel = rgbd_data[:, :, 3]
        
        aligned_rgb = cv2.resize(rgb_channels, 
                               (self.rgb_shape[1], self.rgb_shape[0]),
                               interpolation=cv2.INTER_LINEAR)
        
        aligned_depth = cv2.resize(depth_channel,
                                 (self.rgb_shape[1], self.rgb_shape[0]),
                                 interpolation=cv2.INTER_LINEAR)
        
        # Combine channels
        aligned_rgbd = np.zeros((*self.rgb_shape, 4), dtype=rgbd_data.dtype)
        aligned_rgbd[:, :, :3] = aligned_rgb
        aligned_rgbd[:, :, 3] = aligned_depth
        
        return aligned_rgbd
        
    def extract_object_depth(self, rgbd_aligned: np.ndarray, 
                           image_id: int,
                           category_name: str) -> Dict[str, np.ndarray]:
        """Extract depth data for specific object category"""
        # Get object mask using COCO handler
        mask = self.coco_handler.create_category_mask(
            image_id, 
            category_name, 
            self.rgb_shape
        )
        
        # Extract depth data
        depth_data = rgbd_aligned[:, :, 3].copy()
        masked_depth = np.zeros_like(depth_data)
        masked_depth[mask > 0] = depth_data[mask > 0]
        
        return {
            'mask': mask,
            'depth': masked_depth,
            'category': category_name
        }
