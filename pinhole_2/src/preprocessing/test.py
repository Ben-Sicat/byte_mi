# src/preprocessing/test.py

import os
import sys
import numpy as np
import cv2

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.preprocessing.preprocess import PreprocessingPipeline
from src.utils.utils import load_rgbd_image, create_segmentation_mask, load_coco_data

def test_load_rgbd_image(pipeline):
    rgbd_filename = 'Pairone.png'
    rgbd_image = load_rgbd_image(os.path.join(pipeline.image_input_dir, rgbd_filename))
    assert rgbd_image.shape[2] == 4, f"Expected 4-channel RGBD image, got {rgbd_image.shape[2]} channels"
    print(f"RGBD image shape: {rgbd_image.shape}")
    print(f"RGB min: {rgbd_image[:,:,:3].min()}, max: {rgbd_image[:,:,:3].max()}")
    print(f"Depth min: {rgbd_image[:,:,3].min()}, max: {rgbd_image[:,:,3].max()}")
    print(f"Unique depth values: {np.unique(rgbd_image[:,:,3])}")

def test_create_segmentation_mask(pipeline):
    coco_data = load_coco_data(pipeline.segmentation_file)
    mask = create_segmentation_mask(2, coco_data)  # Using image_id 2 for 'Pair1_png'
    assert mask.shape == (480, 640), f"Expected mask shape (480, 640), got {mask.shape}"
    print(f"Segmentation mask shape: {mask.shape}")
    print(f"Unique mask values: {np.unique(mask)}")

def test_preprocessing_pipeline():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'output'))
    pipeline = PreprocessingPipeline(data_dir, output_dir)

    print("Testing load_rgbd_image function:")
    test_load_rgbd_image(pipeline)

    print("\nTesting create_segmentation_mask function:")
    test_create_segmentation_mask(pipeline)

    print("\nRunning full preprocessing pipeline:")
    pipeline.run()

    # Check if output files are created
    rgb_filenames = [
        'Pair1_png.rf.9a41eaba847f2815f37ffd3e13598fc6.jpg',
        'Pairtwo_png.rf.e23749dcf6644b0a2e561634554a5009.jpg',
        'Pair3_png.rf.984a166a90eb4fb2fc2ea9a4e5a882f4.jpg'
    ]
    
# In the test_preprocessing_pipeline function
    for rgb_filename in rgb_filenames:
        upscaled_rgbd_path = os.path.join(output_dir, 'upscaled', f'{os.path.splitext(rgb_filename)[0]}_upscaled_rgbd.npy')
        assert os.path.exists(upscaled_rgbd_path), f"Upscaled RGBD file not found: {upscaled_rgbd_path}"
        
    visualization_path = os.path.join(output_dir, 'preprocessing_visualization.png')
    assert os.path.exists(visualization_path), f"Visualization file not found: {visualization_path}"

    print("All tests passed successfully!")

if __name__ == "__main__":
    test_preprocessing_pipeline()
