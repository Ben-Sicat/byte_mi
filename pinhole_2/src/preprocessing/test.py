# src/preprocessing/test.py

import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.preprocessing.preprocess import PreprocessingPipeline
from src.utils.utils import load_rgbd_image, create_segmentation_mask, load_coco_data

def test_load_rgbd_image(pipeline):
    """Test loading of RGBD image"""
    rgbd_path = "/home/ben/cThesis/pinhole_2/data/image_input"
    rgbd_files = [f for f in os.listdir(rgbd_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not rgbd_files:
        raise ValueError(f"No RGBD images found in {rgbd_path}")
    
    rgbd_filename = rgbd_files[0]
    full_rgbd_path = os.path.join(rgbd_path, rgbd_filename)
    rgbd_image = load_rgbd_image(full_rgbd_path)
    
    assert rgbd_image is not None, f"Failed to load RGBD image from {full_rgbd_path}"
    assert rgbd_image.shape[2] == 4, f"Expected 4-channel RGBD image, got {rgbd_image.shape[2]} channels"
    
    print(f"RGBD image shape: {rgbd_image.shape}")
    print(f"RGB min: {rgbd_image[:,:,:3].min()}, max: {rgbd_image[:,:,:3].max()}")
    print(f"Depth min: {rgbd_image[:,:,3].min()}, max: {rgbd_image[:,:,3].max()}")
    print(f"Unique depth values: {np.unique(rgbd_image[:,:,3])}")

def test_create_segmentation_mask(pipeline):
    """Test creation of segmentation mask"""
    segmentation_path = "/home/ben/cThesis/pinhole_2/data/train_1/_annotations.coco.json"
    coco_data = load_coco_data(segmentation_path)
    
    if not coco_data['images']:
        raise ValueError("No images found in COCO annotations")
    
    image_id = coco_data['images'][0]['id']
    mask = create_segmentation_mask(image_id, coco_data)
    
    assert mask is not None, "Failed to create segmentation mask"
    assert len(mask.shape) == 2, f"Expected 2D mask, got shape {mask.shape}"
    
    print(f"Segmentation mask shape: {mask.shape}")
    print(f"Unique mask values: {np.unique(mask)}")
    print(f"Number of unique objects: {len(np.unique(mask)) - 1}")  # -1 to exclude background

def test_preprocessing_pipeline():
    """Test the full preprocessing pipeline"""
    data_dir = "/home/ben/cThesis/pinhole_2/data"
    output_dir = "/home/ben/cThesis/pinhole_2/output"
    
    assert os.path.exists(data_dir), f"Data directory not found: {data_dir}"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # initialize pipeline
        pipeline = PreprocessingPipeline(data_dir, output_dir)
        
        # test rgbd loading
        print("\nTesting RGBD image loading...")
        test_load_rgbd_image(pipeline)
        
        # test segmentation mask 
        print("\nTesting segmentation mask creation...")
        test_create_segmentation_mask(pipeline)
        
        # run pipeline
        print("\nRunning full preprocessing pipeline:")
        pipeline.run()
        
        # verify 
        coco_data = load_coco_data(pipeline.segmentation_file)
        image_filename = coco_data['images'][0]['file_name']
        base_filename = os.path.splitext(image_filename)[0]
        
        # check output files
        segmented_depth_path = os.path.join(output_dir, 'processed', f'{base_filename}_segmented_depth.npy')
        visualization_path = os.path.join(output_dir, f'{base_filename}_visualization.png')
        
        assert os.path.exists(segmented_depth_path), f"Segmented depth file not found: {segmented_depth_path}"
        assert os.path.exists(visualization_path), f"Visualization file not found: {visualization_path}"
        
        # load and verify depth 
        depth_data = np.load(segmented_depth_path)
        assert depth_data is not None and depth_data.size > 0, "Empty depth data file"
        
        print("\nAll tests passed successfully!")
        print(f"Output files created in: {output_dir}")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        raise

if __name__ == "__main__":
    test_preprocessing_pipeline()
