# src/utils/visualization.py

import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def visualize_depth(depth_image):
    depth_min, depth_max = np.min(depth_image), np.max(depth_image)
    if depth_min != depth_max:
        normalized_depth = (depth_image - depth_min) / (depth_max - depth_min)
    else:
        normalized_depth = np.zeros_like(depth_image)
    return plt.cm.viridis(normalized_depth)

def visualize_preprocessing_steps(rgb_image, original_rgbd, upscaled_rgbd, segmentation_mask, noise_reduced_depth, output_dir):
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Preprocessing Pipeline Visualization', fontsize=16)

    # High-res RGB
    axs[0, 0].imshow(rgb_image)
    axs[0, 0].set_title('High-res RGB')
    axs[0, 0].axis('off')

    # Original low-res RGBD (color part)
    axs[0, 1].imshow(original_rgbd[:,:,:3].astype(np.uint8))
    axs[0, 1].set_title('Original low-res RGBD (color)')
    axs[0, 1].axis('off')

    # Segmentation Mask
    axs[0, 2].imshow(segmentation_mask, cmap='tab10')
    axs[0, 2].set_title('Segmentation Mask')
    axs[0, 2].axis('off')

    # Upscaled RGBD (color part)
    axs[1, 0].imshow(upscaled_rgbd[:,:,:3].astype(np.uint8))
    axs[1, 0].set_title('Upscaled RGBD (color)')
    axs[1, 0].axis('off')

    # Upscaled Depth
    depth_vis = visualize_depth(upscaled_rgbd[:,:,3])
    axs[1, 1].imshow(depth_vis)
    axs[1, 1].set_title(f'Upscaled Depth (min: {upscaled_rgbd[:,:,3].min():.2f}, max: {upscaled_rgbd[:,:,3].max():.2f})')
    axs[1, 1].axis('off')

    # Upscaled RGBD (false color)
    upscaled_rgbd_vis = np.copy(upscaled_rgbd[:,:,:3])
    upscaled_rgbd_vis[:,:,2] = upscaled_rgbd[:,:,3]  # Replace blue channel with depth
    axs[1, 2].imshow(upscaled_rgbd_vis.astype(np.uint8))
    axs[1, 2].set_title('Upscaled RGBD (false color)')
    axs[1, 2].axis('off')

    # Original Depth
    original_depth_vis = visualize_depth(original_rgbd[:,:,3])
    axs[2, 0].imshow(original_depth_vis)
    axs[2, 0].set_title(f'Original Depth (min: {original_rgbd[:,:,3].min():.2f}, max: {original_rgbd[:,:,3].max():.2f})')
    axs[2, 0].axis('off')

    # Noise Reduced Depth
    noise_reduced_depth_vis = visualize_depth(noise_reduced_depth)
    axs[2, 1].imshow(noise_reduced_depth_vis)
    axs[2, 1].set_title(f'Noise Reduced Depth (min: {noise_reduced_depth.min():.2f}, max: {noise_reduced_depth.max():.2f})')
    axs[2, 1].axis('off')

    # Overlay segmentation on upscaled RGBD
    overlay = overlay_segmentation_on_rgbd(upscaled_rgbd, segmentation_mask)
    axs[2, 2].imshow(overlay[:,:,:3].astype(np.uint8))
    axs[2, 2].set_title('Segmentation on Upscaled RGBD')
    axs[2, 2].axis('off')

    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'preprocessing_visualization.png')
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Visualization saved to: {output_path}")

def overlay_segmentation_on_rgbd(upscaled_rgbd, segmentation_mask):
    # Resize segmentation mask to match upscaled RGBD
    resized_mask = cv2.resize(segmentation_mask, (upscaled_rgbd.shape[1], upscaled_rgbd.shape[0]), 
                              interpolation=cv2.INTER_NEAREST)
    
    # Create a color map for segmentation
    cmap = plt.get_cmap('tab10')
    seg_colors = cmap(resized_mask / np.max(resized_mask))[:,:,:3]
    
    # Create a blended image
    alpha = 0.3  # Adjust this for segmentation transparency
    blended = upscaled_rgbd[:,:,:3] * (1-alpha) + seg_colors * alpha * 255
    
    # Add the depth channel back
    return np.dstack((blended, upscaled_rgbd[:,:,3]))
