import cv2
import numpy as np

""" 
    The goal of this file is to match the resolution of the RGB Image
    since the RGB Depth Image is in a lower resolution
"""

def upscale_depth(rgbd_image, target_shape):
    """
        
        args:
        rgbd_image (numpy.ndarray): RGBD image with depth as the 4th channel
        target_shape (tuple): Desired output shape (height, width)
        
        returns:
        numpy.ndarray: RGBD image with upscaled depth
    
    """
def upscale_depth(rgbd_image, target_shape):
    """
    Upscale the RGBD image to match the target shape while preserving aspect ratio.
    
    Args:
    rgbd_image (numpy.ndarray): RGBD image with depth as the 4th channel
    target_shape (tuple): Desired output shape (height, width)
    
    Returns:
    numpy.ndarray: RGBD image with upscaled depth
    """
    if rgbd_image.shape[2] != 4:
        raise ValueError("Expected RGBD image with 4 channels")
    
    # Calculate scaling factor
    scale = min(target_shape[0] / rgbd_image.shape[0], target_shape[1] / rgbd_image.shape[1])
    
    # Calculate new dimensions
    new_height = int(rgbd_image.shape[0] * scale)
    new_width = int(rgbd_image.shape[1] * scale)
    
    # Upscale RGB
    upscaled_rgb = cv2.resize(rgbd_image[:,:,:3], (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Upscale depth
    upscaled_depth = cv2.resize(rgbd_image[:,:,3], (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    
    # Create padded image
    padded_image = np.zeros((*target_shape, 4), dtype=rgbd_image.dtype)
    
    # Calculate padding
    pad_y = (target_shape[0] - new_height) // 2
    pad_x = (target_shape[1] - new_width) // 2
    
    # Place upscaled image in the center
    padded_image[pad_y:pad_y+new_height, pad_x:pad_x+new_width, :3] = upscaled_rgb
    padded_image[pad_y:pad_y+new_height, pad_x:pad_x+new_width, 3] = upscaled_depth
    
    return padded_image
def align_segmentation_mask(mask, rgbd_shape):

    """
        args:
        mask (numpy.ndarray): Segmentation mask
        rgbd_shape (tuple): shape of rgbd image


        return:
        numpy_ndarry: Aligned Segmentation mask
    """
    
    if mask.shape[:2] != rgbd_shape[:2]:
        return cv2.resize(mask,(rgbd_shape[1],rgbd_shape[0]), interpolation=cv2.INTER_NEAREST)


    return mask



















