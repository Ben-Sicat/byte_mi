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
    if rgbd_image.shape[2] != 4:
        raise ValueError("Expected RGBD image with 4 channels")
    
    h, w = rgbd_image.shape[:2]
    target_h, target_w = target_shape

    # calculate scaling factor while maintaining aspect ratio for the image upscaling not the scaling factor for th pinhole
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # depth channels 
    rgb_resized = cv2.resize(rgbd_image[:,:,:3], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    depth_resized = cv2.resize(rgbd_image[:,:,3], (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # create a new upscaled image 
    upscaled_rgbd = np.zeros((*target_shape, 4), dtype=rgbd_image.dtype)
    
    #  padding
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2

    # put to center para maangas
    upscaled_rgbd[pad_h:pad_h+new_h, pad_w:pad_w+new_w, :3] = rgb_resized
    upscaled_rgbd[pad_h:pad_h+new_h, pad_w:pad_w+new_w, 3] = depth_resized

    return upscaled_rgbd
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



















