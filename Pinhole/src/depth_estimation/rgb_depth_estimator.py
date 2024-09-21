import numpy as np
import cv2

class RGBDDepthEstimator:
    def __init__(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def align_depth_to_color(self, color_image, depth_image):
        # this depends on your specific RGBD camera need help @paulReyes
        pass

    def estimate_depth(self, color_image, depth_image):
        # Align depth to color if necessary
        aligned_depth = self.align_depth_to_color(color_image, depth_image)
        
        # Convert depth image to real-world units (e.g., millimeters)
        # The exact conversion depends on your RGBD camera specifications
        depth_scale = 1  # Replace with actual scale factor
        depth_mm = aligned_depth.astype(float) * depth_scale
        
        return depth_mm

    def get_point_cloud(self, color_image, depth_image):
        depth = self.estimate_depth(color_image, depth_image)
        rows, cols = depth.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
        
        z = depth
        x = (c - self.camera_matrix[0, 2]) * z / self.camera_matrix[0, 0]
        y = (r - self.camera_matrix[1, 2]) * z / self.camera_matrix[1, 1]
        
        return np.dstack((x, y, z))

    def filter_point_cloud(self, point_cloud, min_depth, max_depth):
        mask = (point_cloud[:,:,2] >= min_depth) & (point_cloud[:,:,2] <= max_depth)
        return point_cloud[mask], mask

