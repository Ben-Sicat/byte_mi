import plotly.graph_objects as go
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class Visualizer3D:
    def __init__(self, intrinsic_params: Dict):
        """Initialize visualizer with camera parameters"""
        self.pixel_size = intrinsic_params['pixel_size']
        self.focal_length = intrinsic_params['focal_length']
        self.principal_point = intrinsic_params['principal_point']

    def create_3d_surface(self, 
                         depth_map: np.ndarray,
                         masks: Dict[str, np.ndarray],
                         plate_height: float,
                         output_path: str):
        """
        Create 3D visualization of depth map with colored masks for different objects
        
        Args:
            depth_map: 2D depth map array
            masks: Dictionary of masks for each object {'object_name': mask_array}
            plate_height: Reference plate height
            output_path: Path to save the HTML file
        """
        try:
            rows, cols = depth_map.shape
            y, x = np.mgrid[0:rows, 0:cols]
            
            # Create figure
            fig = go.Figure()
            
            # Colors for different objects
            colors = {
                'plate': 'lightgray',
                'rice': 'orange',
                'egg': 'yellow',
                'background': 'blue'
            }
            
            # Add surface for each object
            for obj_name, mask in masks.items():
                if obj_name in colors:
                    z_values = depth_map.copy()
                    z_values[~mask] = np.nan  # Make non-object points transparent
                    
                    fig.add_trace(go.Surface(
                        x=x * self.pixel_size,
                        y=y * self.pixel_size,
                        z=z_values,
                        name=obj_name,
                        showscale=False,
                        colorscale=[[0, colors[obj_name]], [1, colors[obj_name]]],
                        opacity=0.7
                    ))
            
            # Add background points
            background_mask = ~np.any(list(masks.values()), axis=0)
            if np.any(background_mask):
                z_values = depth_map.copy()
                z_values[~background_mask] = np.nan
                
                fig.add_trace(go.Surface(
                    x=x * self.pixel_size,
                    y=y * self.pixel_size,
                    z=z_values,
                    name='background',
                    showscale=False,
                    colorscale=[[0, colors['background']], [1, colors['background']]],
                    opacity=0.3
                ))
            
            # Update layout
            fig.update_layout(
                title='3D Depth Visualization',
                scene=dict(
                    xaxis_title='X (cm)',
                    yaxis_title='Y (cm)',
                    zaxis_title='Z (cm)',
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    ),
                    aspectmode='data'
                ),
                width=1000,
                height=800
            )
            
            # Add reference plane at plate height
            x_plane = np.array([0, cols * self.pixel_size])
            y_plane = np.array([0, rows * self.pixel_size])
            X_plane, Y_plane = np.meshgrid(x_plane, y_plane)
            Z_plane = np.full_like(X_plane, plate_height)
            
            fig.add_trace(go.Surface(
                x=X_plane,
                y=Y_plane,
                z=Z_plane,
                showscale=False,
                colorscale=[[0, 'red'], [1, 'red']],
                opacity=0.3,
                name='plate_reference'
            ))
            
            # Save figure
            fig.write_html(output_path)
            logger.info(f"3D visualization saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating 3D visualization: {str(e)}")
            raise
