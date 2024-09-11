"""
    This is where we'll gett all instrinsic and extrinsic values and other 
    camera calibration functions to relate to the reference object
"""
import numpy as np 
import cv2

class CameraCalibration:
    def __init__(self):
        self.camera_matrix = None
        self.dist_coeffs = None


    def calibrate(self, image, reference_object_points):

