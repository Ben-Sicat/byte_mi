import numpy as np
import json
import cv2
import plotly.graph_objects as go
from scipy.stats import mode
import traceback
import os

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
    """Calculate camera intrinsic parameters"""
    f_x = focal_length_mm * (image_width_pixels / sensor_width_mm)
    f_y = f_x
    c_x = image_width_pixels / 2
    c_y = image_width_pixels / 2
    return {'f_x': f_x, 'f_y': f_y, 'c_x': c_x, 'c_y': c_y}

def pixel_to_world(u, v, depth_value, intrinsic_params, pixel_size):
    """Convert pixel coordinates to world coordinates"""
    f_x = intrinsic_params['f_x']
    f_y = intrinsic_params['f_y']
    c_x = intrinsic_params['c_x']
    c_y = intrinsic_params['c_y']

    Z = depth_value
    X = ((u - c_x) * Z / f_x) * pixel_size
    Y = ((v - c_y) * Z / f_y) * pixel_size

    return X, Y, Z

def create_binary_mask(segmentation_points, image_shape):
    """Create binary mask from segmentation points"""
    mask = np.zeros(image_shape, dtype=np.uint8)
    points = np.array(segmentation_points).reshape(-1, 2)
    cv2.fillPoly(mask, [points.astype(np.int32)], 1)
    return mask

def estimate_volume_from_mask(depth_map, segmentation_mask, intrinsic_params, pixel_size, plate_height=1.5):
    """Estimate volume using pinhole camera model and 3D reconstruction"""
    try:
        mask = segmentation_mask > 0
        h, w = depth_map.shape
        volume = 0.0
        
        depths = depth_map[mask] - plate_height
        positive_depths = depths[depths > 0]
        
        if len(positive_depths) > 0:
            base_area = np.sum(mask) * (pixel_size ** 2)
            
            # Calculate average height
            avg_height = np.mean(positive_depths)
            max_height = np.max(positive_depths)
            
            #  volume using column 
            for v in range(h):
                for u in range(w):
                    if mask[v, u]:
                        depth_value = depth_map[v, u] - plate_height
                        if depth_value > 0:
                            X, Y, Z = pixel_to_world(u, v, depth_value, intrinsic_params, pixel_size)
                            volume += depth_value * (pixel_size ** 2)
            
            print(f"\nVolume Calculation Statistics:")
            print(f"Base Area (cm²): {base_area:.2f}")
            print(f"Average Height above plate (cm): {avg_height:.2f}")
            print(f"Max Height above plate (cm): {max_height:.2f}")
            print(f"Calculated Volume (cm³): {volume:.2f}")
            
        return volume
            
    except Exception as e:
        print(f"Error in volume estimation: {str(e)}")
        traceback.print_exc()
        return 0.0
def visualize_3d_points(depth_map, intrinsic_params, segmentation_mask, category_name, pixel_size, plate_height=1.5):
    """Visualize 3D point cloud with adjusted heights"""
    points_3d = []
    h, w = segmentation_mask.shape

    for v in range(h):
        for u in range(w):
            if segmentation_mask[v, u] == 1:
                depth_value = depth_map[v, u]
                if depth_value > plate_height:  
                    adjusted_depth = depth_value - plate_height
                    X, Y, Z = pixel_to_world(u, v, adjusted_depth, intrinsic_params, pixel_size)
                    points_3d.append((X, Y, Z))

    if points_3d:
        points_3d = np.array(points_3d)
        
        fig = go.Figure(data=[go.Scatter3d(
            x=points_3d[:, 0],
            y=points_3d[:, 1],
            z=points_3d[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=points_3d[:, 2],  # height / Z
                colorscale='Viridis',
                showscale=True
            )
        )])
        
        fig.update_layout(
            title=f'3D Visualization of {category_name}',
            scene=dict(
                xaxis_title='X (cm)',
                yaxis_title='Y (cm)',
                zaxis_title='Z (cm)',
                aspectmode='data'
            )
        )
        
        return fig
    return None
def calculate_plate_volume(diameter, height):
    """Calculate reference plate volume"""
    radius = diameter / 2
    volume = np.pi * (radius ** 2) * height
    return volume
if __name__ == "__main__":
    # Define absolute paths
    processed_dir = '/home/ben/cThesis/pinhole_2/output/processed'
    coco_file = '/home/ben/cThesis/pinhole_2/data/train_1/_annotations.coco.json'
    output_dir = '/home/ben/cThesis/pinhole_2/output'
    
    # Find the first .npy file in output/processed
    depth_files = [f for f in os.listdir(processed_dir) if f.endswith('_segmented_depth.npy')]
    
    if not depth_files:
        raise ValueError(f"No depth files found in {processed_dir}")
    
    depth_filename = depth_files[0]
    print(f"Processing depth file: {depth_filename}")
    
    # Load depth map
    depth_map = np.load(os.path.join(processed_dir, depth_filename))
    segmentation_data = load_coco_annotations(coco_file)

    # Camera parameters
    focal_length_mm = 7.0
    sensor_width_mm = 4.8
    image_width_pixels = depth_map.shape[1]
    pixel_size = 0.5  

    intrinsic_params = calculate_intrinsic_params(focal_length_mm, sensor_width_mm, image_width_pixels)

    # Plate parameters
    plate_diameter = 25.5  
    plate_height = 1.5    
    
    # Process plate first for calibration
    plate_segmentation_mask = create_binary_mask(segmentation_data[1]['points'], depth_map.shape)
    plate_volume = calculate_plate_volume(plate_diameter, plate_height)
    estimated_plate_volume = estimate_volume_from_mask(
        depth_map, plate_segmentation_mask, intrinsic_params, pixel_size, plate_height
    )
    
    scaling_factor = plate_volume / estimated_plate_volume if estimated_plate_volume > 0 else 1.0
    print(f"\nPlate Calibration:")
    print(f"Actual Plate Volume: {plate_volume:.2f} cm³")
    print(f"Estimated Plate Volume: {estimated_plate_volume:.2f} cm³")
    print(f"Scaling Factor: {scaling_factor:.4f}")

    # Process each segmented object
    for object_id, data in segmentation_data.items():
        category_name = data['category']
        segmentation_mask = create_binary_mask(data['points'], depth_map.shape)
        
        raw_volume = estimate_volume_from_mask(
            depth_map, segmentation_mask, intrinsic_params, pixel_size, plate_height
        )
        adjusted_volume = (raw_volume * scaling_factor) - 1
        vol_cup = (adjusted_volume / 236.588) - 1
        
        print(f"\nResults for '{category_name}':")
        print(f"Raw Volume: {raw_volume:.2f} cm³")
        print(f"Calibrated Volume: {adjusted_volume:.2f} cm³")
        print(f"Volume in Cups: {vol_cup:.2f} cups")
        
        # Update output path for visualizations
        fig = visualize_3d_points(
            depth_map, 
            intrinsic_params, 
            segmentation_mask, 
            category_name, 
            pixel_size, 
            plate_height
        )
        fig.write_html(os.path.join(output_dir, f'3d_points_visualization_{category_name}.html'))
