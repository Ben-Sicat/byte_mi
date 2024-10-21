import os
import json
import cv2
import numpy as np

def load_rgbd_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    if img.shape[2] != 4:
        raise ValueError(f"Expected 4-channel RGBD image, got {img.shape[2]} channels")
    
    print(f"Loaded RGBD image shape: {img.shape}")
    print(f"Depth channel min: {img[:,:,3].min()}, max: {img[:,:,3].max()}")
    print(f"Unique depth values: {np.unique(img[:,:,3])}")
    
    return img

def load_coco_data(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def create_segmentation_mask(image_id, coco_data):
    image_info = next((img for img in coco_data['images'] if img['id'] == image_id), None)
    if image_info is None:
        raise ValueError(f"No image found with id {image_id}")
    
    mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)
    
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
    
    print(f"Number of annotations for image {image_id}: {len(annotations)}")
    
    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    for ann in annotations:
        category_id = ann['category_id']
        for segmentation in ann['segmentation']:
            pts = np.array(segmentation).reshape((-1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [pts], color=category_id)
    
    unique_ids = np.unique(mask)
    print("Categories in the image:")
    for id in unique_ids:
        if id != 0:  # Skip background
            print(f"  ID {id}: {category_id_to_name.get(id, 'Unknown')}")
    
    return mask
def get_corresponding_rgbd_filename(rgb_filename):
    mapping = {
        'normal_pair4_png.rf.fa99eaa222e8d4acfcfb6483600dda01.jpg': 'depth_pair4.png',
    }
    return mapping.get(rgb_filename)

def ensure_directory_exists(directory):
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)

def align_segmentation_mask(mask, target_shape):
    """Resize segmentation mask to match target shape."""
    if mask.shape[:2] != target_shape:
        return cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask

def save_depth_image(depth_data, output_path, as_uint8=True):
    """
    Save depth data as an image file.
    
    Args:
    depth_data (numpy.ndarray): 2D array of depth values
    output_path (str): Path to save the image
    as_uint8 (bool): If True, convert depth to uint8 before saving
    """
    if as_uint8:
        # normalizedepth to 0 255 range
        depth_min = np.min(depth_data)
        depth_max = np.max(depth_data)
        if depth_min != depth_max:
            depth_normalized = ((depth_data - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        else:
            depth_normalized = np.zeros_like(depth_data, dtype=np.uint8)
    else:
        depth_normalized = depth_data

    cv2.imwrite(output_path, depth_normalized)

def load_depth_image(input_path, as_float=True):
    """
    Load a depth image file.
    
    Args:
    input_path (str): Path to the depth image file
    as_float (bool): If True, convert depth to float32
    
    Returns:
    numpy.ndarray: 2D array of depth values
    """
    depth_image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    
    if depth_image is None:
        raise FileNotFoundError(f"Could not load depth image at {input_path}")
    
    if as_float:
        depth_image = depth_image.astype(np.float32) / 255.0
    
    return depth_image

def visualize_depth(depth_image, output_path):
    """
    Create a color visualization of a depth image and save it.
    
    Args:
    depth_image (numpy.ndarray): 2D array of depth values
    output_path (str): Path to save the visualization
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(depth_image, cmap='viridis')
    plt.colorbar(label='Depth')
    plt.title('Depth Visualization')
    plt.axis('off')
    plt.savefig(output_path)
    plt.close()
