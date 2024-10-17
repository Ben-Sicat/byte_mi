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
def visualize_preprocessing_steps(rgb_image, original_rgbd, upscaled_rgbd, segmentation_mask, segmented_data, output_dir, filename):
    num_objects = len(segmented_data)
    rows = 4 + num_objects
    fig, axs = plt.subplots(rows, 3, figsize=(20, 7 * rows))
    fig.suptitle('Preprocessing Pipeline Visualization', fontsize=16)

    # Ensure axs is always a 2D array
    if rows == 1:
        axs = axs.reshape(1, -1)

    # First row: RGB, RGBD, and Segmentation Mask
    axs[0, 0].imshow(rgb_image)
    axs[0, 0].set_title('High-res RGB')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(original_rgbd[:,:,:3])
    axs[0, 1].set_title('Original RGBD (color)')
    axs[0, 1].axis('off')

    axs[0, 2].imshow(segmentation_mask, cmap='tab10')
    axs[0, 2].set_title('Segmentation Mask')
    axs[0, 2].axis('off')

    # Second row: Upscaled RGBD visualizations
    axs[1, 0].imshow(upscaled_rgbd[:,:,:3])
    axs[1, 0].set_title('Upscaled RGBD (color)')
    axs[1, 0].axis('off')

    # Combined segmented RGBD
    combined_segmented_rgbd = np.zeros_like(upscaled_rgbd)
    for obj_data in segmented_data.values():
        combined_segmented_rgbd = np.maximum(combined_segmented_rgbd, obj_data['rgbd'])
    axs[1, 1].imshow(combined_segmented_rgbd[:,:,:3])
    axs[1, 1].set_title('Segmented RGBD (color)')
    axs[1, 1].axis('off')

    # Leave the last subplot of the second row empty for now
    axs[1, 2].axis('off')

    # Third row: Color histograms of upscaled RGBD
    for i, color in enumerate(['red', 'green', 'blue']):
        axs[2, i].hist(upscaled_rgbd[:,:,i].ravel(), bins=256, range=(0, 255), color=color, alpha=0.7)
        axs[2, i].set_title(f'Upscaled RGBD {color.capitalize()} Channel Histogram')
        axs[2, i].set_xlabel('Color Value')
        axs[2, i].set_ylabel('Frequency')

    # Fourth row: Color histograms of segmented objects
    for i, color in enumerate(['red', 'green', 'blue']):
        all_object_colors = np.concatenate([obj_data['colors'][:,i] for obj_data in segmented_data.values()])
        axs[3, i].hist(all_object_colors, bins=256, range=(0, 255), color=color, alpha=0.7)
        axs[3, i].set_title(f'Segmented Objects {color.capitalize()} Channel Histogram')
        axs[3, i].set_xlabel('Color Value')
        axs[3, i].set_ylabel('Frequency')

    # Individual object visualizations
    for i, (obj_id, obj_data) in enumerate(segmented_data.items()):
        row = 4 + i
        
        # RGB part
        axs[row, 0].imshow(obj_data['rgb'])
        axs[row, 0].set_title(f'{obj_data["name"]} RGB')
        axs[row, 0].axis('off')
        
        # RGBD part
        axs[row, 1].imshow(obj_data['rgbd'][:,:,:3])
        axs[row, 1].set_title(f'{obj_data["name"]} RGBD (color)')
        axs[row, 1].axis('off')
        
        # Color histogram for this object
        for j, color in enumerate(['red', 'green', 'blue']):
            axs[row, 2].hist(obj_data['colors'][:,j], bins=256, range=(0, 255), color=color, alpha=0.7)
        axs[row, 2].set_title(f'{obj_data["name"]} Color Histogram')
        axs[row, 2].set_xlabel('Color Value')
        axs[row, 2].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
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
