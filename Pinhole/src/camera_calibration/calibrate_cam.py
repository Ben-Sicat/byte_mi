# src/camera_calibration/calibrate_cam.py

import numpy as np
import cv2
from scipy.optimize import minimize

class TrayBasedCalibrator:
    def __init__(self, plate_dimentions, image_size):
        """
        Initialize the calibrator with tray specifications and image size.
        """
        self.pate_length, self.plate_width, self.plate_height = plate_dimentions
        self.image_size = image_size
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvec = None
        self.tvec = None

    def calibrate(self, section_mask):
        full_plate_mask = np.sum(section_mask, axis = 0) > 0
        # extract tray contour
        contours, _ = cv2.findContours(tray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        plate_contour = max(contours, key=cv2.contourArea)

        # rotated rectangle 
        rect = cv2.minAreaRect(plate_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # initial guess for intrinsic parameters
        focal_length = self._estimate_focal_length(rect[1])
        principal_point = (self.image_size[1] / 2, self.image_size[0] / 2)
        self.camera_matrix = np.array([
            [focal_length, 0, principal_point[0]],
            [0, focal_length, principal_point[1]],
            [0, 0, 1]
        ])
        self.dist_coeffs = np.zeros(5)

        # 3D points of the tray corners
        object_points = self._generate_3d_tray_points()

        # 2D points from the tray corners
        image_points = box.astype(np.float32)

        # Optimize camera parameters
        self._optimize_parameters(object_points, image_points)

        return self.camera_matrix, self.dist_coeffs, self.rvec, self.tvec

    def _estimate_focal_length(self, rect_size):
        pixel_size = max(rect_size)
        return (pixel_size * 1) / self.tray_specs['overall_width']

    def _generate_3d_plate_points(self):
        l, w = self.plate_length / 2, self.plate_width / 2
        return np.array([
            [-l, -w, 0],
            [l, -w, 0],
            [l, w, 0],
            [-l, w, 0]
        ], dtype=np.float32)

    def _optimize_parameters(self, object_points, image_points):
        def objective(params):
            fx, fy, cx, cy, k1, k2, p1, p2, k3, tx, ty, tz, rx, ry, rz = params
            
            self.camera_matrix[0, 0] = fx
            self.camera_matrix[1, 1] = fy
            self.camera_matrix[0, 2] = cx
            self.camera_matrix[1, 2] = cy
            self.dist_coeffs = np.array([k1, k2, p1, p2, k3])
            
            self.rvec = np.array([rx, ry, rz])
            self.tvec = np.array([tx, ty, tz])
            
            projected_points, _ = cv2.projectPoints(object_points, self.rvec, self.tvec, self.camera_matrix, self.dist_coeffs)
            
            error = np.sum((image_points - projected_points.squeeze())**2)
            return error

        initial_params = [
            self.camera_matrix[0, 0], self.camera_matrix[1, 1],
            self.camera_matrix[0, 2], self.camera_matrix[1, 2],
            0, 0, 0, 0, 0, # distortion coeff
            0, 0, 1,        # translation
            0, 0, 0         # rotation
        ]

        result = minimize(objective, initial_params, method='Powell')
        
        fx, fy, cx, cy, k1, k2, p1, p2, k3, tx, ty, tz, rx, ry, rz = result.x
        self.camera_matrix[0, 0] = fx
        self.camera_matrix[1, 1] = fy
        self.camera_matrix[0, 2] = cx
        self.camera_matrix[1, 2] = cy
        self.dist_coeffs = np.array([k1, k2, p1, p2, k3])
        self.rvec = np.array([rx, ry, rz])
        self.tvec = np.array([tx, ty, tz])

    def get_intrinsic_parameters(self):
        return {
            'camera_matrix': self.camera_matrix,
            'dist_coeffs': self.dist_coeffs
        }

    def get_extrinsic_parameters(self):
        return {
            'rvec': self.rvec,
            'tvec': self.tvec
        }
#try test 
def calibrate_camera(tray_mask, image_size):
    tray_specs = {
        'overall_width': 300,  # in mm
        'overall_height': 250,  # in mm
        'compartments': [
            {'width': 150, 'height': 150},
            {'width': 150, 'height': 100},
            {'width': 75, 'height': 75},
            {'width': 75, 'height': 75},
            {'width': 75, 'height': 75}
        ]
    }
    
    calibrator = TrayBasedCalibrator(tray_specs, image_size)
    camera_matrix, dist_coeffs, rvec, tvec = calibrator.calibrate(tray_mask)
    
    return calibrator.get_intrinsic_parameters(), calibrator.get_extrinsic_parameters()
