import numpy as np
import cv2
import json
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CocoHandler:
    """Utility class for handling COCO format annotations"""
    def __init__(self, annotation_file: str):
        self.annotations = self._load_annotations(annotation_file)
        self.categories = {cat['id']: cat['name'] 
                          for cat in self.annotations['categories']}
        
    def _load_annotations(self, file_path: str) -> Dict:
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading COCO annotations: {str(e)}")
            raise
            
    def get_image_annotations(self, image_id: int) -> List[Dict]:
        return [ann for ann in self.annotations['annotations'] 
                if ann['image_id'] == image_id]
    
    def get_category_id(self, category_name: str) -> int:
        """Get category ID from name"""
        for cat_id, name in self.categories.items():
            if name == category_name:
                return cat_id
        raise ValueError(f"Category {category_name} not found")
    
    def create_mask(self, annotation: Dict, shape: Tuple[int, int]) -> np.ndarray:
        """Create binary mask from single annotation"""
        mask = np.zeros(shape, dtype=np.uint8)
        for segmentation in annotation['segmentation']:
            points = np.array(segmentation).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [points], 1)
        return mask

    def create_category_mask(self, image_id: int, 
                           category_name: str, 
                           shape: Tuple[int, int]) -> np.ndarray:
        """Create mask for specific category"""
        category_id = self.get_category_id(category_name)
        mask = np.zeros(shape, dtype=np.uint8)
        
        annotations = [ann for ann in self.get_image_annotations(image_id)
                      if ann['category_id'] == category_id]
        
        for ann in annotations:
            mask = cv2.bitwise_or(mask, self.create_mask(ann, shape))
            
        return mask
