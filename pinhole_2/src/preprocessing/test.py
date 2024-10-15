import sys
import os
import json
import numpy as np
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.preprocessing.image_scaling import upscale_depth, align_segmentation_mask
from src.preprocessing.noise_reduction import reduce_depth_noise
from src.utils.visualization import visualize_preprocessing_steps

# Paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
IMAGE_INPUT_DIR = os.path.join(DATA_DIR, 'image_input')
TEST_DIR = os.path.join(DATA_DIR, 'train')
SEGMENTATION_FILE = os.path.join(DATA_DIR, 'train', '_annotations.coco.json')

def load_rgbd_image(filename):
    img = cv2.imread(os.path.join(IMAGE_INPUT_DIR, filename), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {os.path.join(IMAGE_INPUT_DIR, filename)}")
    if img.shape[2] != 4:
        raise ValueError(f"Expected 4-channel RGBD image, got {img.shape[2]} channels")
    print(f"RGBD image shape: {img.shape}")
    print(f"RGB min: {img[:,:,:3].min()}, max: {img[:,:,:3].max()}")
    print(f"Depth min: {img[:,:,3].min()}, max: {img[:,:,3].max()}")
    print(f"Unique depth values: {np.unique(img[:,:,3])}")

    return img

def load_coco_data(json_file):
    """Load COCO format data from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)

def create_segmentation_mask(image_id, coco_data):
    """Create a segmentation mask for a given image from COCO data."""
    image_info = next(img for img in coco_data['images'] if img['id'] == image_id)
    mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)
    
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
    
    for ann in annotations:
        category_id = ann['category_id']
        for segmentation in ann['segmentation']:
            pts = np.array(segmentation).reshape((-1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [pts], color=category_id)  # Use category_id as color

    return mask

def get_corresponding_rgbd_filename(rgb_filename):
    """Get the corresponding RGBD filename for a given RGB filename."""
    mapping = {
        'Pair1_png.rf.9a41eaba847f2815f37ffd3e13598fc6.jpg': 'Pairone.png',
        'Pairtwo_png.rf.e23749dcf6644b0a2e561634554a5009.jpg': 'Pairtwo.png',
        'Pair3_png.rf.984a166a90eb4fb2fc2ea9a4e5a882f4.jpg': 'Pairthree.png'
    }
    return mapping.get(rgb_filename)
########"""
#visualization
########
def test_preprocessing_pipeline():
    print("Starting preprocessing pipeline test")
    
    # Load high-resolution RGB image
    rgb_filename = 'Pair1_png.rf.9a41eaba847f2815f37ffd3e13598fc6.jpg'
    rgb_path = os.path.join(TEST_DIR, rgb_filename)
    print(f"Attempting to load RGB image from: {rgb_path}")
    
    if not os.path.exists(rgb_path):
        print(f"Error: RGB image file not found at {rgb_path}")
        return

    rgb_image = cv2.imread(rgb_path)
    if rgb_image is None:
        print(f"Error: Unable to read RGB image at {rgb_path}")
        return

    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    print(f"RGB image shape: {rgb_image.shape}")

    # Load low-resolution RGBD image
    rgbd_filename = get_corresponding_rgbd_filename(rgb_filename)
    rgbd_path = os.path.join(IMAGE_INPUT_DIR, rgbd_filename)
    print(f"Attempting to load RGBD image from: {rgbd_path}")
    
    if not os.path.exists(rgbd_path):
        print(f"Error: RGBD image file not found at {rgbd_path}")
        return

    original_rgbd = load_rgbd_image(rgbd_filename)
    print(f"Original RGBD shape: {original_rgbd.shape}")

    # Upscale RGBD to match RGB resolution
    upscaled_rgbd = upscale_depth(original_rgbd, (rgb_image.shape[0], rgb_image.shape[1]))
    print(f"Upscaled RGBD shape: {upscaled_rgbd.shape}")

    # Load and create segmentation mask (using the high-res RGB image)
    coco_data = load_coco_data(SEGMENTATION_FILE)
    mask = create_segmentation_mask(2, coco_data)  # Using image_id 2 for 'Pair1_png'
    print(f"Segmentation mask shape: {mask.shape}")

    # ensures mask shape matches rgb image shape
    if mask.shape[:2] != rgb_image.shape[:2]:
        mask = cv2.resize(mask, (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Reduce noise in depth
    original_depth = original_rgbd[:,:,3].astype(np.float32)
    noise_reduced_depth = reduce_depth_noise(original_depth)
    upscaled_noise_reduced_depth = cv2.resize(noise_reduced_depth, (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_LINEAR)
    print(f"Noise reduced depth shape: {upscaled_noise_reduced_depth.shape}")

    # Visualize results
    visualize_preprocessing_steps(rgb_image, original_rgbd, upscaled_rgbd, mask, noise_reduced_depth, output_dir)
    print("Preprocessing pipeline test completed")

# Make sure these paths are correctly set at the top of your file
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
TEST_DIR = os.path.join(DATA_DIR, 'train')
IMAGE_INPUT_DIR = os.path.join(DATA_DIR, 'image_input')
SEGMENTATION_FILE = os.path.join(DATA_DIR, 'train', '_annotations.coco.json')
output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'output')
os.makedirs(output_dir, exist_ok=True)

if __name__ == "__main__":
    test_preprocessing_pipeline()
