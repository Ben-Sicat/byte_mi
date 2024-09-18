import numpy as np
import cv2

class PlateBasedCalibrator:
    def __init__(self, plate_dimensions, image_size):
        """
        Initialize the calibrator with plate specifications and image size.
        """
        self.plate_length, self.plate_width, self.plate_height = plate_dimensions
        self.image_size = image_size
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvec = None
        self.tvec = None

    def calibrate(self, plate_mask):
        """
        Calibrate the camera using the plate mask.
        """
        if plate_mask.shape[:2] != self.image_size:
            raise ValueError("Mask size does not match the specified image size.")

        # Extract plate contour
        contours, _ = cv2.findContours(plate_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No contours found in the plate mask.")
        plate_contour = max(contours, key=cv2.contourArea)

        # Get rotated rectangle
        rect = cv2.minAreaRect(plate_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Generate 3D points of the plate corners
        object_points = self._generate_3d_plate_points()

        # Use 2D points from the plate corners
        image_points = box.astype(np.float32)

        # Estimate camera parameters
        ret, self.camera_matrix, self.dist_coeffs, self.rvec, self.tvec = cv2.calibrateCamera(
            [object_points], [image_points], self.image_size, None, None
        )

        if not ret:
            raise RuntimeError("Camera calibration failed.")

        return self.camera_matrix, self.dist_coeffs, self.rvec, self.tvec

    def _generate_3d_plate_points(self):
        l, w = self.plate_length / 2, self.plate_width / 2
        return np.array([
            [-l, -w, 0],
            [l, -w, 0],
            [l, w, 0],
            [-l, w, 0]
        ], dtype=np.float32)

    def get_intrinsic_parameters(self):
        if self.camera_matrix is None or self.dist_coeffs is None:
            raise RuntimeError("Camera not calibrated. Call calibrate() first.")
        return {
            'camera_matrix': self.camera_matrix,
            'dist_coeffs': self.dist_coeffs
        }

    def get_extrinsic_parameters(self):
        if self.rvec is None or self.tvec is None:
            raise RuntimeError("Camera not calibrated. Call calibrate() first.")
        return {
            'rvec': self.rvec,
            'tvec': self.tvec
        }

    def project_3d_to_2d(self, points_3d):
        if self.camera_matrix is None or self.dist_coeffs is None:
            raise RuntimeError("Camera not calibrated. Call calibrate() first.")
        points_2d, _ = cv2.projectPoints(points_3d, self.rvec, self.tvec, self.camera_matrix, self.dist_coeffs)
        return points_2d.squeeze()

    def unproject_2d_to_3d(self, points_2d, Z=0):
        if self.camera_matrix is None:
            raise RuntimeError("Camera not calibrated. Call calibrate() first.")
        points_2d = np.array(points_2d, dtype=np.float32)
        points_3d = cv2.undistortPoints(points_2d, self.camera_matrix, self.dist_coeffs)
        points_3d = points_3d.squeeze()
        points_3d = np.column_stack([points_3d, np.ones(len(points_3d))])
        points_3d *= Z
        return points_3d
#
# def calibrate_camera(plate_mask, image_size, plate_dimensions):
#     calibrator = PlateBasedCalibrator(plate_dimensions, image_size)
#     calibrator.calibrate(plate_mask)
#     return calibrator.get_intrinsic_parameters(), calibrator.get_extrinsic_parameters()
