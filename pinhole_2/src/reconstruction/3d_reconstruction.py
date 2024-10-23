import numpy as np
import json
import cv2 
import plotly.graph_objects as go
from scipy.stats import mode

def load_coco_annotations(coco_file):
    """Load COCO annotations and return segmentation data."""
    with open(coco_file, 'r') as f:
        annotations = json.load(f)


    segmentation_data = {}
    
    category_mapping = {cat['id']: cat['name'] for cat in annotations['categories']}

    for annotation in annotations['annotations']:
        object_id = annotation['id']
        segmentation_points = annotation['segmentation'][0]  
        category_id = annotation['category_id']
        category_name = category_mapping.get(category_id, "Unknown")
        segmentation_data[object_id] = {
            'points': segmentation_points,
            'category': category_name
        }

    return segmentation_data

def calculate_intrinsic_params(focal_length_mm, sensor_width_mm, image_width_pixels):
    """Calculate intrinsic camera parameters."""
    f_x = focal_length_mm * (image_width_pixels / sensor_width_mm)
    f_y = f_x  
    c_x = image_width_pixels / 2
    c_y = image_width_pixels / 2
    return {'f_x': f_x, 'f_y': f_y, 'c_x': c_x, 'c_y': c_y}

def pixel_to_world(u, v, depth_value, intrinsic_params, pixel_size):
    """Convert pixel coordinates and depth to world coordinates."""
    f_x = intrinsic_params['f_x']
    f_y = intrinsic_params['f_y']
    c_x = intrinsic_params['c_x']
    c_y = intrinsic_params['c_y']

    Z = depth_value  
    X = ((u - c_x) * Z / f_x) * pixel_size
    Y = ((v - c_y) * Z / f_y) * pixel_size

    return X, Y,Z

def create_binary_mask(segmentation_points, image_shape):
    """Create a binary mask from segmentation points."""
    mask = np.zeros(image_shape, dtype=np.uint8)
    points = np.array(segmentation_points).reshape(-1, 2)
    cv2.fillPoly(mask, [points.astype(np.int32)], 1)


    return mask

def estimate_volume_from_mask(depth_map, segmentation_mask, pixel_size, fixed_depth=1.34):
    """estimate volume using a fixed depth value and area of the mask.
    
    this is to be replaced with the std out data from the preprocessing piepline

    """
    valid_depth_values = depth_map[segmentation_mask > 0]
    
    pixel_area = np.sum(segmentation_mask > 0) * (pixel_size ** 2)  
    volume = pixel_area * fixed_depth  

    print(f"Valid Depth Values: {valid_depth_values[:5]}")  
    print(f"Mask Area (cm²): {pixel_area:.2f}")  
    print(f"Fixed Depth (Z): {fixed_depth:.2f} cm")  
    print(f"Estimated Volume (before scaling): {volume:.2f} cm³")  


    return volume

def visualize_3d_points(depth_map, intrinsic_params, segmentation_mask, category_name, pixel_size):
    points_3d = []

    h, w = segmentation_mask.shape
    for v in range(h):
        for u in range(w):
            if segmentation_mask[v, u] == 1:  # shit inside segments
                depth_value = depth_map[v, u]
                if depth_value > 0:  
                    X, Y, Z = pixel_to_world(u, v, depth_value, intrinsic_params, pixel_size)
                    points_3d.append((X, Y, Z))

    points_3d = np.array(points_3d)

    x = points_3d[:, 0]
    y = points_3d[:, 1]
    z = points_3d[:, 2]

    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=2))])
    fig.update_layout(title=f'3D Visualization of {category_name}',
                      scene=dict(xaxis_title='X (cm)', yaxis_title='Y (cm)', zaxis_title='Z (cm)'))

    fig.write_html(f'output/3d_points_visualization_{category_name}.html')

def calculate_plate_volume(diameter, height):
    """Calculate the volume of the plate."""
    radius = diameter / 2
    volume = np.pi * (radius ** 2) * height 
    return volume
if __name__ == "__main__":
    # Load the depth map
    depth_map = np.load('output/processed/normal_pair4_png.rf.fa99eaa222e8d4acfcfb6483600dda01_segmented_depth.npy')

    # Load segmentation points from COCO annotations
    coco_file = 'data/train_1/_annotations.coco.json'
    segmentation_data = load_coco_annotations(coco_file)

    # Camera settings
    focal_length_mm = 5.0  # Example focal length in mm (adjust based on your camera)
    sensor_width_mm = 4.8  # Example sensor width in mm (adjust based on your camera)
    image_width_pixels = depth_map.shape[1]  # Width of the depth map image

    # Calculate intrinsic camera parameters
    intrinsic_params = calculate_intrinsic_params(focal_length_mm, sensor_width_mm, image_width_pixels)

    # Assuming each pixel corresponds to 0.5 cm (adjust as needed)
    pixel_size = 0.5  # Size of each pixel in centimeters (cm)

    # Calculate the volume of the plate using known dimensions
    plate_diameter = 25.5  # cm
    plate_height = 1.5  # cm
    plate_volume = calculate_plate_volume(plate_diameter, plate_height)
    print(f"Calculated Plate Volume: {plate_volume:.2f} cm³")

    # Estimate the volume for the plate using the depth map
    plate_segmentation_mask = create_binary_mask(segmentation_data[1]['points'], depth_map.shape)  # Assuming plate is object ID 1
    estimated_plate_volume = estimate_volume_from_mask(depth_map, plate_segmentation_mask, pixel_size, fixed_depth=1.34)

    # Check the estimated plate volume before calculating scaling factor
    print(f"Estimated Plate Volume: {estimated_plate_volume:.2f} cm³")

    # Avoid division by zero when calculating scaling factor
    scaling_factor = plate_volume / estimated_plate_volume if estimated_plate_volume > 0 else 1.0
    print(f"Scaling Factor: {scaling_factor:.4f}")

    # Iterate over each object in the segmentation data
    for object_id, data in segmentation_data.items():
        segmentation_points = data['points']
        category_name = data['category']

        # Create a binary mask for the current object
        segmentation_mask = create_binary_mask(segmentation_points, depth_map.shape)

        # Estimate the volume for the current object using depth values
        volume_cm3 = estimate_volume_from_mask(depth_map, segmentation_mask, pixel_size, fixed_depth=1.34)

        # Apply the scaling factor derived from the plate
        adjusted_volume = volume_cm3 * scaling_factor
        print(f"Estimated Volume for '{category_name}': {adjusted_volume:.2f} cm³")

        # Convert the volume to cups
        volume_cups = adjusted_volume / 236.588
        print(f"Estimated Volume for '{category_name}' in Cups: {volume_cups:.2f} cups")

        # Visualize the 3D points for the segmented object
        visualize_3d_points(depth_map, intrinsic_params, segmentation_mask, category_name, pixel_size)

