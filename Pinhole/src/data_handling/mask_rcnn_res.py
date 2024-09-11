import json
from typing import Dict, List, Any, Tuple

class MaskRCNNResultsHandler:
    def __init__(self, results_path: str):
        self.results_path = results_path
        self.segmentation_data = None
    def load_results(self) -> None:
        with open(self.results_path, 'r') as f:
            self.segmentation_data = json.load(f)


    def get_segmentations(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns:
            Dict[str, Dict[str, Any]]: A dictionary where keys are object names (food items or plate sections)
                                       and values are dictionaries containing 'segmentation', 'section', and 'category'.
        """
        if self.segmentation_data is None:
            raise ValueError("Results not loaded. Call load_results() first.")
        
        segmentations = {}
        for item in self.segmentation_data['annotations']:
            segmentations[item['category_name']] = {
                'segmentation': item['segmentation'],
                'section': item.get('section_name', None), 
                'category': 'food' if item.get('section_name') else 'plate_section'
            }
        
        return segmentations

    def get_plate_sections(self) -> Dict[str, List[List[float]]]:
        """
        Returns:
            Dict[str, List[List[float]]]: A dictionary where keys are plate section names
                                          and values are lists of segmentation points.
        """
        segmentations = self.get_segmentations()
        return {name: data['segmentation'] for name, data in segmentations.items() if data['category'] == 'plate_section'}
    def get_food_items(self) -> Dict[str, Tuple[List[List[float]], str]]:
        """
        Returns:
            Dict[str, Tuple[List[List[float]], str]]: A dictionary where keys are food item names
                                                      and values are tuples of (segmentation points, section name).
        """
        segmentations = self.get_segmentations()
        return {name: (data['segmentation'], data['section']) for name, data in segmentations.items() if data['category'] == 'food'}

    def get_metadata(self) -> Dict[str, Any]:
        """
        Returns:
            Dict[str, Any]: Metadata dictionary.
        """
        if self.segmentation_data is None:
            raise ValueError("Results not loaded. Call load_results() first.")
        
        return {
            'image_size': self.segmentation_data['image']['size'],
            'categories': self.segmentation_data['categories']
        }

    def get_bounding_boxes(self) -> Dict[str, List[float]]:
        """
        Returns:
            Dict[str, List[float]]: A dictionary where keys are object names
                                    and values are bounding box coordinates [x, y, width, height].
        """
        if self.segmentation_data is None:
            raise ValueError("Results not loaded. Call load_results() first.")
        
        return {item['category_name']: item['bbox'] for item in self.segmentation_data['annotations']}
