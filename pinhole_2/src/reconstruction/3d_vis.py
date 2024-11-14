import os
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

def get_latest_depth_file(processed_dir):
    """Find the most recently created depth map file in the processed directory"""
    depth_files = [f for f in os.listdir(processed_dir) if f.endswith('_segmented_depth.npy')]
    
    if not depth_files:
        raise FileNotFoundError(f"No depth map files found in {processed_dir}")
    
    # Get full paths and sort by creation time
    depth_files_with_time = [
        (f, os.path.getctime(os.path.join(processed_dir, f)))
        for f in depth_files
    ]
    latest_file = sorted(depth_files_with_time, key=lambda x: x[1], reverse=True)[0][0]
    
    return latest_file

def visualize_depth_map(processed_dir='output/processed', output_dir='output/visualization'):
    """Create 3D visualization of the depth map"""
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get the latest depth map file
        latest_file = get_latest_depth_file(processed_dir)
        depth_map_path = os.path.join(processed_dir, latest_file)
        
        print(f"Processing depth map: {latest_file}")
        
        # Load the depth map
        depth_map = np.load(depth_map_path)
        
        # Create meshgrid
        h, w = depth_map.shape
        xx, yy = np.meshgrid(np.arange(0, w), np.arange(0, h))
        
        # Create surface plot
        fig = go.Figure(data=[
            go.Surface(
                z=depth_map,
                x=xx,
                y=yy,
                colorscale='Viridis',
                colorbar=dict(
                    title='Depth (cm)',
                    titleside='right'
                )
            )
        ])
        
        # Customize layout
        fig.update_layout(
            title=dict(
                text=f'3D Surface Reconstruction from Depth Map<br><sub>{latest_file}</sub>',
                x=0.5,
                y=0.95
            ),
            scene=dict(
                xaxis_title='X (Pixels)',
                yaxis_title='Y (Pixels)',
                zaxis_title='Depth (cm)',
                zaxis=dict(range=[0, 10]),  # Adjust range as needed
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)  # Adjust camera position
                )
            )
        )
        
        # Adjust aspect ratio
        fig.update_scenes(aspectratio=dict(x=1, y=1, z=0.2))
        
        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f'3d_surface_reconstruction_{timestamp}.html'
        output_path = os.path.join(output_dir, output_filename)
        
        # Save visualization
        fig.write_html(output_path)
        print(f"Visualization saved as '{output_path}'")
        
        # Additional information about the depth map
        print("\nDepth Map Statistics:")
        print(f"Shape: {depth_map.shape}")
        print(f"Min depth: {depth_map.min():.2f} cm")
        print(f"Max depth: {depth_map.max():.2f} cm")
        print(f"Mean depth: {depth_map.mean():.2f} cm")
        print(f"Standard deviation: {depth_map.std():.2f} cm")
        
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        raise

if __name__ == "__main__":
    visualize_depth_map()
