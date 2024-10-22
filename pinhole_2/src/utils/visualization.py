import matplotlib
matplotlib.use('Agg')  # set Agg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def visualize_depth(depth_image):
    depth_min, depth_max = np.nanmin(depth_image), np.nanmax(depth_image)
    if depth_min != depth_max:
        normalized_depth = (depth_image - depth_min) / (depth_max - depth_min)
    else:
        normalized_depth = np.zeros_like(depth_image)
    return plt.cm.viridis(normalized_depth)
def visualize_preprocessing_steps(rgb_image, rgbd_image, normalized_rgbd, segmentation_mask, segmented_data, output_dir, output_path, normalized_plate_color, combined_segmented_depth):
    fig, axs = plt.subplots(2, 3, figsize=(20, 15))
    fig.suptitle('Preprocessing Pipeline Visualization', fontsize=16)

    # Original RGB Image
    axs[0, 0].imshow(rgb_image)
    axs[0, 0].set_title('Original RGB Image')
    axs[0, 0].axis('off')

    # RGBD Image (showing RGB part)
    axs[0, 1].imshow(rgbd_image[:,:,:3])
    axs[0, 1].set_title('RGBD Image (RGB part)')
    axs[0, 1].axis('off')

    # Segmentation Mask
    axs[0, 2].imshow(segmentation_mask, cmap='tab20')
    axs[0, 2].set_title('Segmentation Mask')
    axs[0, 2].axis('off')

    # Normalized RGBD (showing RGB part)
    axs[1, 0].imshow(normalized_rgbd[:,:,:3])
    axs[1, 0].set_title('Normalized RGBD (RGB part)')
    axs[1, 0].axis('off')

    # Calibrated Depth
    depth_plot = axs[1, 1].imshow(combined_segmented_depth, cmap='viridis')
    axs[1, 1].set_title('Segmented Depth')
    axs[1, 1].axis('off')
    plt.colorbar(depth_plot, ax=axs[1, 1], label='Depth (cm)')

    # Segmented Objects with Depth
    axs[1, 2].imshow(rgb_image)
    depth_mask = combined_segmented_depth > 0
    axs[1, 2].imshow(np.ma.masked_where(~depth_mask, combined_segmented_depth), cmap='jet', alpha=0.5)
    axs[1, 2].set_title('Segmented Objects with Depth')
    axs[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Visualization saved to: {output_path}")

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
