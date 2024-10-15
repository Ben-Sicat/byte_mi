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

    if rgbd_image.shape[2] == 4:
        rgb = rgbd_image[:,:,:3]
        depth = rgbd_image[:,:,3]
    else:
        raise ValueError("Expected RGBD image with 4 channels")

    upscaled_rgb = cv2.resize(rgb, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
    upscaled_depth = cv2.resize(depth, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)


    return np.dstack((upscaled_rgb,upscaled_depth))

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



















