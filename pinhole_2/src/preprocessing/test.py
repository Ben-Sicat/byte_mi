import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import cv2

from src.preprocessing.image_scaling import upscale_depth, align_segmentation_mask
from src.preprocessing.noise_reduction import reduce_depth_noise

# paths
DATA_DIR = 'data'
IMAGE_INPUT_DIR = os.path.join(DATA_DIR, 'image_input')
SEGMENTATION_FILE = os.path.join(DATA_DIR, 'train', '_annotations.coco.json')

def load_rgbd_image(filename):
    """Load RGBD image from a PNG file."""
    img = cv2.imread(os.path.join(IMAGE_INPUT_DIR, filename), cv2.IMREAD_UNCHANGED)
    if img.shape[2] != 4:  # if its just RGB.... add  dummy d channel
        raise ValueError(f"Expected 4-channel RGBD image, got {img.shape[2]} channels")
    return img

def load_rgb_image(filename):
    """ load RGB image"""
    return cv2.imread(os.path.join(IMAGE_INPUT_DIR, filename), cv2.IMREAD_COLOR)

def load_coco_data(json_file):
    """load COCO format data"""
    with open(json_file, 'r') as f:
        return json.load(f)

def load_segmentation_mask(image_id, coco_data):
    """ create the segmentation mask from coco format data"""
    image_info = next(img for img in coco_data['images'] if img['id'] == image_id)
    mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)

    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]

    for ann in annotations:
        for segmentation in ann['segmentation']:
            pts = np.array(segmentation).reshape((-1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [pts], color=1)

    return mask

def get_corresponding_rgbd_filename(rgb_filename):
    """get the corresponding rgbdfilename for a given rgb """
    mapping = {
        'Pair1_png.rf.9a41eaba847f2815f37ffd3e13598fc6.jpg': 'Pairone.png',
        'Pairtwo_png.rf.e23749dcf6644b0a2e561634554a5009.jpg': 'Pairtwo.png',
        'Pair3_png.rf.984a166a90eb4fb2fc2ea9a4e5a882f4.jpg': 'Pairthree.png'
    }
    return mapping.get(rgb_filename)
def test_upscale_depth():
    rgb_filename = 'Pair1_png.rf.9a41eaba847f2815f37ffd3e13598fc6.jpg'
    rgbd_filename = get_corresponding_rgbd_filename(rgb_filename)
    rgbd_image = load_rgbd_image(rgbd_filename)
    print(f"Original RGBD shape: {rgbd_image.shape}")
    target_shape = (rgbd_image.shape[0] * 2, rgbd_image.shape[1] * 2)
    print(f"Target shape: {target_shape}")
    upscaled_rgbd = upscale_depth(rgbd_image, target_shape)
    print(f"Upscaled RGBD shape: {upscaled_rgbd.shape}")
    assert upscaled_rgbd.shape[:2] == target_shape, "Upscaling failed"
    print("Upscale depth test passed")
def test_align_segmentation_mask():
    coco_data = load_coco_data(SEGMENTATION_FILE)
    rgb_filename = 'Pair1_png.rf.9a41eaba847f2815f37ffd3e13598fc6.jpg'
    rgbd_filename = get_corresponding_rgbd_filename(rgb_filename)
    rgbd_image = load_rgbd_image(rgbd_filename)
    rgb_image = load_rgb_image(rgb_filename)
    mask = load_segmentation_mask(2, coco_data)  # Using image_id 2 for 'Pair1_png'
    aligned_mask = align_segmentation_mask(mask, rgbd_image.shape)
    assert aligned_mask.shape[:2] == rgbd_image.shape[:2], "Mask alignment failed"
    print("Align segmentation mask test passed")

def test_reduce_depth_noise():
    rgb_filename = 'Pair1_png.rf.9a41eaba847f2815f37ffd3e13598fc6.jpg'
    rgbd_filename = get_corresponding_rgbd_filename(rgb_filename)
    rgbd_image = load_rgbd_image(rgbd_filename)
    depth = rgbd_image[:, :, 3]  # Access depth channel
    reduced_noise_depth = reduce_depth_noise(depth)
    assert reduced_noise_depth.shape == depth.shape, "Noise reduction changed image shape"
    # error over here fix this
    assert not np.array_equal(reduced_noise_depth, depth), "Noise reduction had no effect"
    print("Reduce depth noise test passed")

if __name__ == "__main__":
    test_upscale_depth()
    test_align_segmentation_mask()
    test_reduce_depth_noise()

