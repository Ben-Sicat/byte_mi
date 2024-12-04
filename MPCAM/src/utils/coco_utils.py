import numpy as np
import cv2
import json
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CocoHandler:
    """Utility class for handling COCO format annotations"""
    def __init__(self, annotation_file: str):
        self.annotations = self._load_annotations(annotation_file)
        self.categories = {cat['id']: cat['name'] 
                          for cat in self.annotations['categories']}
        
        # Create filename to image_id mapping
        self.filename_to_id = {}
        for img in self.annotations['images']:
            # Strip extension and any extra suffixes
            base_name = img['file_name'].split('_png')[0]
            self.filename_to_id[base_name] = img['id']
            
        logger.info(f"Loaded categories: {list(self.categories.values())}")
        logger.info(f"Loaded image mappings: {self.filename_to_id}")
        
    def _load_annotations(self, file_path: str) -> Dict:
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading COCO annotations: {str(e)}")
            raise
            
    def get_image_id(self, frame_id: str) -> int:
        """Get COCO image ID from frame ID"""
        filename = f"rgb_frame_{frame_id}"
        if filename not in self.filename_to_id:
            logger.error(
                f"No image ID found for {filename}. "
                f"Available files: {list(self.filename_to_id.keys())}"
            )
            raise ValueError(f"Image ID not found for frame {frame_id}")
        return self.filename_to_id[filename]
            
    def get_image_annotations(self, frame_id: str) -> List[Dict]:
        """Get annotations for specific image"""
        try:
            image_id = self.get_image_id(frame_id)
            annotations = [ann for ann in self.annotations['annotations'] 
                         if ann['image_id'] == image_id]
            
            if annotations:
                logger.info(f"Found {len(annotations)} annotations for image {frame_id}")
            else:
                logger.warning(f"No annotations found for image {frame_id}")
                
            return annotations
            
        except Exception as e:
            logger.error(f"Error getting annotations: {str(e)}")
            raise
    
    def get_category_id(self, category_name: str) -> int:
        """Get category ID from name"""
        for cat_id, name in self.categories.items():
            if name.lower() == category_name.lower():
                return cat_id
        available_categories = list(self.categories.values())
        raise ValueError(
            f"Category '{category_name}' not found. "
            f"Available categories are: {available_categories}"
        )
    
    def create_mask(self, annotation: Dict, shape: Tuple[int, int]) -> np.ndarray:
        """Create binary mask from single annotation"""
        mask = np.zeros(shape, dtype=np.uint8)
        
        if not annotation.get('segmentation'):
            logger.error(f"No segmentation data in annotation: {annotation}")
            return mask
            
        try:
            for segmentation in annotation['segmentation']:
                points = np.array(segmentation).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [points], 1)
                
            if not np.any(mask):
                logger.warning("Created mask is empty")
            else:
                logger.info(f"Created mask with {np.sum(mask)} positive pixels")
                
            return mask
            
        except Exception as e:
            logger.error(f"Error creating mask: {str(e)}")
            return mask

    def create_category_mask(self, frame_id: str, 
                           category_name: str, 
                           shape: Tuple[int, int]) -> np.ndarray:
        """Create mask for specific category"""
        try:
            category_id = self.get_category_id(category_name)
            mask = np.zeros(shape, dtype=np.uint8)
            
            annotations = self.get_image_annotations(frame_id)
            category_annotations = [
                ann for ann in annotations 
                if ann['category_id'] == category_id
            ]
            
            if not category_annotations:
                logger.warning(
                    f"No annotations found for category '{category_name}' "
                    f"in frame {frame_id}"
                )
                return mask
                
            for ann in category_annotations:
                ann_mask = self.create_mask(ann, shape)
                mask = cv2.bitwise_or(mask, ann_mask)
                
            if not np.any(mask):
                logger.warning(f"Final mask for category '{category_name}' is empty")
            else:
                logger.info(
                    f"Created mask for category '{category_name}' with "
                    f"{np.sum(mask)} positive pixels"
                )
                
            return mask
            
        except Exception as e:
            logger.error(f"Error creating category mask: {str(e)}")
            return np.zeros(shape, dtype=np.uint8)
        
    def visualize_mask(self, mask: np.ndarray, save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize a binary mask and optionally save it.
        
        Args:
            mask: Binary mask to visualize
            save_path: Optional path to save visualization
            
        Returns:
            np.ndarray: Visualization image
        """
        if not np.any(mask):
            logger.warning("Mask is empty - no visualization created")
            return np.zeros((*mask.shape, 3), dtype=np.uint8)
            
        # Create color visualization
        viz = np.zeros((*mask.shape, 3), dtype=np.uint8)
        viz[mask > 0] = [0, 255, 0]  # Green for masked areas
        
        # Add contours
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(viz, contours, -1, (255, 255, 255), 2)
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
            logger.info(f"Saved mask visualization to {save_path}")
            
        return viz
