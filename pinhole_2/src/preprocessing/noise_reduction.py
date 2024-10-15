import cv2
import numpy as np


def reduce_depth_noise(depth_image, method='median'):
    """
        args:
        depth_image (numpy.ndarray): depth depth_image
        method (str) : noise reduction method ('bilateral' or 'median')

        return:
        numpy.ndarray: Noise reduced dpeth depth_image

    """
    depth_image = depth_image.astype(np.float32)
    if method == 'bilateral':
        return cv2.bilateralFilter(depth_image, 9,75,75)
    elif method== 'median':
        return cv2.medianBlur(depth_image,5)
    else:
        raise ValueError('unsupported noise reduction method')
        
