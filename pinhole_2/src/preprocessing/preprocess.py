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
from src.preprocessing.image_scaling import upscale_depth, align_segmentation_mask
from src.preprocessing.noise_reduction import reduce_depth_noise
from src.utils.visualization import visualize_preprocessing_steps, visualize_depth

class PreprocessingPipeline:
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.image_input_dir = os.path.join(data_dir, 'image_input')
        self.train_dir = os.path.join(data_dir, 'train')
        self.segmentation_file = os.path.join(data_dir, 'train', '_annotations.coco.json')
        
        # Create output directories
        ensure_directory_exists(output_dir)
        self.upscaled_dir = os.path.join(output_dir, 'upscaled')
        ensure_directory_exists(self.upscaled_dir)

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

            # Load low-resolution RGBD image
            rgbd_path = os.path.join(self.image_input_dir, rgbd_filename)
            original_rgbd = load_rgbd_image(rgbd_path)
            print(f"Loaded RGBD image shape: {original_rgbd.shape}")
            print(f"Loaded RGBD depth min: {original_rgbd[:,:,3].min()}, max: {original_rgbd[:,:,3].max()}")

            # Upscale RGBD to match RGB resolution
            upscaled_rgbd = upscale_depth(original_rgbd, (rgb_image.shape[0], rgb_image.shape[1]))
            print(f"Upscaled RGBD shape: {upscaled_rgbd.shape}")
            print(f"Upscaled RGBD depth min: {upscaled_rgbd[:,:,3].min()}, max: {upscaled_rgbd[:,:,3].max()}")

            # Load and create segmentation mask
            coco_data = load_coco_data(self.segmentation_file)
            mask = create_segmentation_mask(image_id, coco_data)
            print(f"Created segmentation mask shape: {mask.shape}")

            # Ensure mask shape matches RGB image shape
            mask = align_segmentation_mask(mask, rgb_image.shape[:2])
            print(f"Aligned segmentation mask shape: {mask.shape}")

            # Reduce noise in depth
            original_depth = original_rgbd[:,:,3].astype(np.float32)
            noise_reduced_depth = reduce_depth_noise(original_depth)
            upscaled_noise_reduced_depth = cv2.resize(noise_reduced_depth, (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_LINEAR)
            print(f"Noise reduced depth min: {upscaled_noise_reduced_depth.min()}, max: {upscaled_noise_reduced_depth.max()}")

            # Save upscaled RGBD image as numpy array (full color + depth)
            upscaled_rgbd_path = os.path.join(self.upscaled_dir, f'{os.path.splitext(rgb_filename)[0]}_upscaled_rgbd.npy')
            np.save(upscaled_rgbd_path, upscaled_rgbd)
            print(f"Saved upscaled RGBD to: {upscaled_rgbd_path}")

            upscaled_rgbd_vis_path = os.path.join(self.upscaled_dir, f'{os.path.splitext(rgb_filename)[0]}_upscaled_rgbd_vis.png')
            rgbd_vis = upscaled_rgbd[:,:,:3].astype(np.uint8)  # Use only the RGB channels for visualization
            cv2.imwrite(upscaled_rgbd_vis_path, cv2.cvtColor(rgbd_vis, cv2.COLOR_RGB2BGR))
            print(f"Saved upscaled RGBD visualization to: {upscaled_rgbd_vis_path}")

            # Visualize results
            visualize_preprocessing_steps(rgb_image, original_rgbd, upscaled_rgbd, mask, noise_reduced_depth, self.output_dir)
            print(f"Saved preprocessing visualization to: {self.output_dir}")

            return upscaled_rgbd, mask


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
            upscaled_rgbd, mask = self.process_image(rgb_filename, image_id)
            print(f"Processed {rgb_filename}. Upscaled RGBD shape: {upscaled_rgbd.shape}, Mask shape: {mask.shape}")
            print(f"Saved upscaled RGBD and visualizations to {self.upscaled_dir}")
            print("---")
