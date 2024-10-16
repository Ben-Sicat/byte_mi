# src/utils/visualization.py

import matplotlib
matplotlib.use('Agg')  # set Agg
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
def visualize_preprocessing_steps(rgb_image, rgbd_image, estimated_depth, calibrated_depth, segmentation_mask, segmented_depths, output_dir):
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Preprocessing Pipeline Visualization', fontsize=16)

    # High-res RGB
    axs[0, 0].imshow(rgb_image)
    axs[0, 0].set_title('High-res RGB')
    axs[0, 0].axis('off')

    # Original RGBD (color part)
    axs[0, 1].imshow(rgbd_image[:,:,:3])
    axs[0, 1].set_title('Original RGBD (color)')
    axs[0, 1].axis('off')

    # Segmentation Mask
    axs[0, 2].imshow(segmentation_mask, cmap='tab10')
    axs[0, 2].set_title('Segmentation Mask')
    axs[0, 2].axis('off')

    # Estimated Depth (low-res)
    im_estimated = axs[1, 0].imshow(estimated_depth, cmap='viridis')
    axs[1, 0].set_title('Estimated Depth (low-res)')
    axs[1, 0].axis('off')
    plt.colorbar(im_estimated, ax=axs[1, 0], label='Depth (cm)')

    # Upscaled and Calibrated Depth
    im_calibrated = axs[1, 1].imshow(calibrated_depth, cmap='viridis')
    axs[1, 1].set_title('Calibrated Depth (high-res)')
    axs[1, 1].axis('off')
    plt.colorbar(im_calibrated, ax=axs[1, 1], label='Depth (cm)')

    # Segmented Depths
    combined_segmented_depth = np.zeros_like(calibrated_depth)
    for obj_depth in segmented_depths.values():
        combined_segmented_depth = np.nan_to_num(combined_segmented_depth) + np.nan_to_num(obj_depth)
    im_segmented = axs[1, 2].imshow(combined_segmented_depth, cmap='viridis')
    axs[1, 2].set_title('Segmented Depths')
    axs[1, 2].axis('off')
    plt.colorbar(im_segmented, ax=axs[1, 2], label='Depth (cm)')

    # Individual object depths
    for i, (obj_id, obj_depth) in enumerate(segmented_depths.items()):
        if i >= 3:  # Only show up to 3 individual objects
            break
        im_obj = axs[2, i].imshow(obj_depth, cmap='viridis')
        axs[2, i].set_title(f'Object {obj_id} Depth')
        axs[2, i].axis('off')
        plt.colorbar(im_obj, ax=axs[2, i], label='Depth (cm)')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/preprocessing_visualization.png")
    plt.close(fig)

def overlay_segmentation_on_rgbd(upscaled_rgbd, segmentation_mask):
    # Resize segmentation mask to match upscaled RGBD if necessary
    if upscaled_rgbd.shape[:2] != segmentation_mask.shape:
        segmentation_mask = cv2.resize(segmentation_mask, (upscaled_rgbd.shape[1], upscaled_rgbd.shape[0]), 
                                       interpolation=cv2.INTER_NEAREST)
    
    # Create a color map for segmentation
    cmap = plt.get_cmap('tab10')
    seg_colors = cmap(segmentation_mask / np.max(segmentation_mask))[:,:,:3]
    
    # Create a blended image
    alpha = 0.3  # Adjust this for segmentation transparency
    blended = upscaled_rgbd[:,:,:3] * (1-alpha) + seg_colors * alpha * 255
    
    # Add the depth channel back if it exists
    if upscaled_rgbd.shape[2] == 4:
        return np.dstack((blended, upscaled_rgbd[:,:,3]))
    else:
        return blended
