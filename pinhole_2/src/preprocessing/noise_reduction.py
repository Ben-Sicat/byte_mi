import cv2
import numpy as np


def reduce_depth_noise(depth_image, method='bilateral'):
    """
        args:
        depth_image (numpy.ndarray): depth depth_image
        method (str) : noise reduction method ('bilateral' or 'median')

        return:
        numpy.ndarray: Noise reduced dpeth depth_image

    """
    print(f"Noise reduction input depth min: {depth_image.min()}, max: {depth_image.max()}")
    
    if method == 'bilateral':
        result = cv2.bilateralFilter(depth_image, 5, 50, 50)
    elif method == 'median':
        result = cv2.medianBlur(depth_image, 5)
    else:
        raise ValueError('Unsupported noise reduction method')
    
    print(f"Noise reduction output depth min: {result.min()}, max: {result.max()}")
    return result
