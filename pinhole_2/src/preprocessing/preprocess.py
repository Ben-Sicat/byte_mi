import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.utils.utils import (
    load_rgbd_image,
    load_coco_data,
    create_segmentation_mask,
    get_corresponding_rgbd_filename,
    ensure_directory_exists
)
from src.preprocessing.image_scaling import align_segmentation_mask
from src.utils.visualization import visualize_preprocessing_steps

class PreprocessingPipeline:
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.image_input_dir = os.path.join(data_dir, 'image_input')
        self.train_dir = os.path.join(data_dir, 'train')
        self.segmentation_file = os.path.join(data_dir, 'train', '_annotations.coco.json')
        
        ensure_directory_exists(output_dir)
        self.processed_dir = os.path.join(output_dir, 'processed')
        ensure_directory_exists(self.processed_dir)
        
        #load data
        self.coco_data = load_coco_data(self.segmentation_file)

        # creat mapping of category
        self.category_id_to_name = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        # Camera setup constants
        self.camera_height = 33  
        self.plate_height = 1.5 
        self.plate_depth = 0.7

    def color_to_depth(self, color_image):
        """
        Convert color information to estimated depth values.
        
        Args:
        color_image (numpy.ndarray): RGB image

        Returns:
        numpy.ndarray: Estimated depth values (in cm)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
        
        # Normalize to 0-1 range
        normalized = gray.astype(float) / 255.0
        
        # Invert so that darker colors are further away
        inverted = 1 - normalized
        
        # Scale depth values based on camera setup
        # Assume the brightest points (now 0) are at plate level, and darkest (now 1) are at max depth
        max_depth = self.camera_height - self.plate_height
        depth = inverted * max_depth + self.plate_height
        
        return depth

    def calibrate_depth(self, depth, segmentation_mask, plate_id=5):
        """
        Calibrate depth using the plate as a reference.
        
        Args:
        depth (numpy.ndarray): Estimated depth values
        segmentation_mask (numpy.ndarray): Segmentation mask
        plate_id (int): ID of the plate in the segmentation mask

        Returns:
        numpy.ndarray: Calibrated depth values
        """
        plate_mask = segmentation_mask == plate_id
        plate_depth = depth[plate_mask]
        
        if plate_depth.size == 0:
            print("Warning: No plate found in the image for calibration.")
            return depth
        
        plate_avg_depth = np.mean(plate_depth)
        expected_plate_depth = self.plate_height + self.plate_depth / 2  # Average depth of the plate
        
        # Adjust depth so that the plate has the expected depth
        calibration_factor = expected_plate_depth / plate_avg_depth
        calibrated_depth = depth * calibration_factor
        
        return calibrated_depth
    def upscale_with_padding(self, image, target_shape):
        """
        Upscale the image to target shape while maintaining aspect ratio and using padding.
        
        Args:
        image (numpy.ndarray): Input image to upscale
        target_shape (tuple): Target shape (height, width)

        Returns:
        numpy.ndarray: Upscaled and padded image
        """
        h, w = image.shape[:2]
        target_h, target_w = target_shape

        # Calculate scaling factor
        scale = min(target_h / h, target_w / w)
        
        # Calculate new dimensions
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize the image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create a black canvas of the target shape
        if len(image.shape) == 3:
            padded = np.zeros((target_h, target_w, image.shape[2]), dtype=resized.dtype)
        else:
            padded = np.zeros((target_h, target_w), dtype=resized.dtype)

        # Compute padding
        pad_h = (target_h - new_h) // 2
        pad_w = (target_w - new_w) // 2

        # Place the resized image on the canvas
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized

        return padded
    def process_image(self, rgb_filename, image_id):
        try:
            # Load high-resolution RGB image
            rgb_path = os.path.join(self.train_dir, rgb_filename)
            rgb_image = cv2.imread(rgb_path)
            if rgb_image is None:
                raise FileNotFoundError(f"Could not load RGB image at {rgb_path}")
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            print(f"Loaded RGB image shape: {rgb_image.shape}")

            # Get corresponding RGBD filename
            rgbd_filename = get_corresponding_rgbd_filename(rgb_filename)
            if rgbd_filename is None:
                raise ValueError(f"No corresponding RGBD filename found for {rgb_filename}")

            # Load RGBD image
            rgbd_path = os.path.join(self.image_input_dir, rgbd_filename)
            rgbd_image = load_rgbd_image(rgbd_path)
            print(f"Loaded RGBD image shape: {rgbd_image.shape}")

            # Upscale RGBD image to match RGB resolution
            upscaled_rgbd = self.upscale_with_padding(rgbd_image, rgb_image.shape[:2])
            print(f"Upscaled RGBD shape: {upscaled_rgbd.shape}")

            # Create segmentation mask
            coco_data = load_coco_data(self.segmentation_file)
            segmentation_mask = create_segmentation_mask(image_id, coco_data)
            segmentation_mask = align_segmentation_mask(segmentation_mask, rgb_image.shape[:2])
            print(f"Segmentation mask shape: {segmentation_mask.shape}")
            print(f"Unique values in segmentation mask: {np.unique(segmentation_mask)}")

            # Extract color data for segmented objects
            segmented_data = {}
            for obj_id in np.unique(segmentation_mask):
                if obj_id == 0:  # Assuming 0 is background
                    continue
                object_mask = segmentation_mask == obj_id
                
                # Extract RGB data for the object
                object_rgb = rgb_image.copy()
                object_rgb[~object_mask] = 0
                
                # Extract RGBD color data for the object
                object_rgbd = upscaled_rgbd.copy()
                object_rgbd[~object_mask] = 0
                
                # Get object name from COCO categories
                object_name = self.category_id_to_name.get(obj_id, f"unknown_object_{obj_id}")
                
                # Extract color values for pixels inside the segmentation mask
                object_colors = object_rgbd[object_mask]
                
                segmented_data[obj_id] = {
                    'rgb': object_rgb, 
                    'rgbd': object_rgbd,
                    'colors': object_colors,
                    'name': object_name
                }

            # Visualize results
            vis_filename = f"{os.path.splitext(rgb_filename)[0]}_visualization.png"
            visualize_preprocessing_steps(rgb_image, rgbd_image, upscaled_rgbd, segmentation_mask, segmented_data, self.output_dir, vis_filename)

            return rgb_image, upscaled_rgbd, segmentation_mask, segmented_data
        except Exception as e:
            print(f"Error processing image {rgb_filename}: {str(e)}")
            raise
    def run(self):
        rgb_filenames = [
            'Pair1_png.rf.9a41eaba847f2815f37ffd3e13598fc6.jpg',
            'Pairtwo_png.rf.e23749dcf6644b0a2e561634554a5009.jpg',
            'Pair3_png.rf.984a166a90eb4fb2fc2ea9a4e5a882f4.jpg'
        ]
        image_ids = [2, 0, 1]  # Corresponding image IDs in the COCO dataset

        for rgb_filename, image_id in zip(rgb_filenames, image_ids):
            print(f"Processing {rgb_filename}")
            rgb, depth, mask, segmented_depths = self.process_image(rgb_filename, image_id)
            print(f"Processed {rgb_filename}.")
            print(f"RGB shape: {rgb.shape}, Depth shape: {depth.shape}, Mask shape: {mask.shape}")
            print(f"Number of segmented objects: {len(segmented_depths)}")
            print(f"Saved processed data to {self.processed_dir}")
            print("---")
