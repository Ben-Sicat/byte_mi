"""
    mask rcnn utility functiions
"""

import sys
import os
import logging
import math
import random
import numpy as np
import tensorflow as tf
import scipy
import skimage.color
import skimage.io
import skimage.transform
import urllib.request
import shutil
import warnings


def bounding_boxes(mask):
    """
    from the mask compute the bounding boxes.
    
    so mask is [height, width, num_instances]. Mask pixels are either 1 or 0
    
    return bounding_boxes array [num_instances, (y1, x1, y2, x2)]
    
    """
    # mask.shape is the num of instaces and 4 are the coordinates of bounding box  ok? fuck off
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    
    for i in range (mask.shape[-1]):
        #selects the entire instance of the mask in `i`
        instance_m = instance_m[:, :, i]
        #BOUNDING BOX
        non_zeros_row = np.any(instance_m, axis=0)
        non_zeros_cols= np.any(instance_m, axis=1)   
        
        
        #check if pixels belong in i
        if non_zeros_row.any():
            x1,x2 = np.where(non_zeros_row)[0][[0,-1]]
            y1,y2 = np.where(non_zeros_cols)[0][[0,-1]]
            
            x2 +=1
            y2 +=1
        else:
            
            x1,y1,x2,y2 = 0,0,0,0            
        
        boxes[i] = np.array([y1,x1,y2,x2])
    return boxes.astype(np.int32)

def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas hyper parameters should be based on the dataset so for now
    # we will use the default values
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou

def compute_overlaps(boxes1,boxes2):
    """
        computes the Intersection-over-Union overlaps
        
          Args:
                boxes1: A numpy array of shape [N1, 4]. Each row represents a bounding box: [y1, x1, y2, x2].
                boxes2: A numpy array of shape [N2, 4]. Each row represents a bounding box: [y1, x1, y2, x2].
            Returns:
        A numpy array of shape [N1, N2].
        - Each cell `[i, j]` represents the IoU between box i in boxes1 and box j in boxes2.

    """
    
    assert boxes1.ndim == 2 and boxes1.shape[1] == 4, "boxes1 should have shape [N, 4]"
    assert boxes2.ndim == 2 and boxes2.shape[1] == 4, "boxes2 should have shape [N, 4]"
    
    # area of box
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps

def compute_overlap_mask(mask1, mask2):
    """Computes IoU overlaps: masks: [Height, Width, instances]"""
    
    #if masks are empty return empty array
    if mask1.shape[-1] == 0 or mask2.shape[-1] == 0:
        return np.zeros((mask1.shape[-1], mask2.shape[-1]))
    #flatten
    mask1 = np.reshape(mask1 > .5, (-1, mask1.shape[-1])).astype(np.float32)
    mask2 = np.reshape(mask2 > .5, (-1, mask2.shape[-1])).astype(np.float32)
    area1 = np.sum(mask1, axis=0)
    area2 = np.sum(mask2, axis=0)
    
    #intersection and union
    intersections = np.dot(mask1.T, mask2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union
    
    return overlaps