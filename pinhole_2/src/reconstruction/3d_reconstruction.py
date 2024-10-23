import numpy as np
import json
import cv2
import plotly.graph_objects as go
from scipy.stats import mode

def load_coco_annotations(coco_file):
    """Load COCO annotations."""
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
    """get camera params"""
    f_x = focal_length_mm * (image_width_pixels / sensor_width_mm)
    f_y = f_x


    c_x = image_width_pixels / 2
    c_y = image_width_pixels / 2
    return {'f_x': f_x, 'f_y': f_y, 'c_x': c_x, 'c_y': c_y}

def pixel_to_world(u, v, depth_value, intrinsic_params, pixel_size):
    """convert pixel to world coords."""
    f_x = intrinsic_params['f_x']
    f_y = intrinsic_params['f_y']
    c_x = intrinsic_params['c_x']
    c_y = intrinsic_params['c_y']

    Z = depth_value  

    X = ((u - c_x) * Z / f_x) * pixel_size  # in cm
    Y = ((v - c_y) * Z / f_y) * pixel_size  # in cm

    return X,Y, Z

def create_binary_mask(segmentation_points, image_shape):
    mask = np.zeros(image_shape, dtype=np.uint8)
    points = np.array(segmentation_points).reshape(-1, 2)  
    cv2.fillPoly(mask, [points.astype(np.int32)], 1)  
    return mask

def estimate_volume_from_mask(depth_map, segmentation_mask, pixel_size, fixed_depth=1.34):
    """estimate volume"""
    valid_depth_values = depth_map[segmentation_mask > 0]
    
    pixel_area = np.sum(segmentation_mask > 0) * (pixel_size ** 2)  
    volume = pixel_area * fixed_depth  

    print(f"Mask Area (cm²): {pixel_area:.2f}")  
    print(f"Estimated Volume (before scaling): {volume:.2f} cm³")

    return volume

def visualize_3d_points(depth_map, intrinsic_params, segmentation_mask, category_name, pixel_size):
    """Show 3D points."""
    points_3d = []

    h, w = segmentation_mask.shape

    for v in range(h):
        for u in range(w):
            if segmentation_mask[v, u] == 1:  # only the good points
                depth_value = depth_map[v, u]
                if depth_value > 0:  # valid depth
                    X, Y, Z = pixel_to_world(u, v, depth_value, intrinsic_params, pixel_size)
                    points_3d.append((X, Y, Z))

    points_3d = np.array(points_3d)

    x = points_3d[:, 0]
    y = points_3d[:, 1]
    z = points_3d[:, 2]

    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=2))])
    fig.update_layout(title=f'3D Visualization of {category_name}',scene=dict(xaxis_title='X (cm)',yaxis_title='Y (cm)', zaxis_title='Z (cm)'))

    fig.write_html(f'output/3d_points_visualization_{category_name}.html')

def calculate_plate_volume(diameter, height):
    """Calc the volume of the plate."""
    radius = diameter / 2
    volume = np.pi * (radius ** 2) * height #random shit i got HAHAHAHA 
    return volume

if __name__ == "__main__":
    depth_map = np.load('output/processed/normal_pair4_png.rf.fa99eaa222e8d4acfcfb6483600dda01_segmented_depth.npy')
    coco_file = 'data/train_1/_annotations.coco.json'

    segmentation_data = load_coco_annotations(coco_file)

    focal_length_mm = 7.0  # chatgpt values HAHA
    sensor_width_mm = 4.8  
    image_width_pixels = depth_map.shape[1]  # width of depth map

    intrinsic_params = calculate_intrinsic_params(focal_length_mm, sensor_width_mm, image_width_pixels)
    pixel_size = 0.5


    plate_diameter = 25.5  
    plate_height = 1.5
    plate_volume = calculate_plate_volume(plate_diameter, plate_height)

    plate_segmentation_mask = create_binary_mask(segmentation_data[1]['points'], depth_map.shape)  # plate ID 1
    estimated_plate_volume = estimate_volume_from_mask(depth_map, plate_segmentation_mask, pixel_size, fixed_depth=1.34)
    scaling_factor = plate_volume / estimated_plate_volume if estimated_plate_volume > 0 else 1.0

    for object_id, data in segmentation_data.items():
        segmentation_points = data['points']
        category_name = data['category']

        segmentation_mask = create_binary_mask(segmentation_points, depth_map.shape)

        volume_cm3 = estimate_volume_from_mask(depth_map, segmentation_mask, pixel_size, fixed_depth=1.34)

        adjusted_volume = volume_cm3 * scaling_factor
        print(f"Estimated Volume for '{category_name}': {adjusted_volume:.2f} cm³")

        volume_cups = adjusted_volume / 236.588
        print(f"Estimated Volume for '{category_name}' in Cups: {volume_cups:.2f} cups")

        visualize_3d_points(depth_map, intrinsic_params, segmentation_mask, category_name, pixel_size)

