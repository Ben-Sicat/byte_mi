import plotly.graph_objects as go
import numpy as np
from typing import Dict, Optional, Tuple
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DepthVisualizer:
    """
    Visualization tools for depth data analysis and validation.
    """
    def __init__(self, camera_height: float, plate_diameter: float, plate_height: float):
        """
        Initialize visualizer with camera parameters.
        
        Args:
            camera_height: Height of camera in cm
            plate_diameter: Diameter of reference plate in cm
            plate_height: Height of plate in cm
        """
        self.camera_height = camera_height
        self.plate_diameter = plate_diameter
        self.plate_height = plate_height
        self.expected_plate_distance = camera_height - plate_height
        
    def create_depth_surface(self, depth_map: np.ndarray, 
                           mask: Optional[np.ndarray] = None,
                           title: str = "Depth Surface Plot") -> go.Figure:
        """
        Create interactive 3D surface plot of depth data.
        
        Args:
            depth_map: 2D depth map array
            mask: Optional binary mask to focus on specific region
            title: Plot title
            
        Returns:
            plotly.graph_objects.Figure
        """
        # Apply mask if provided
        if mask is not None:
            depth_data = depth_map.copy()
            depth_data[mask == 0] = np.nan
        else:
            depth_data = depth_map
            
        # Create coordinate grids
        rows, cols = depth_data.shape
        x = np.linspace(0, cols-1, cols)
        y = np.linspace(0, rows-1, rows)
        X, Y = np.meshgrid(x, y)
        
        # Create surface plot
        fig = go.Figure(data=[
            go.Surface(
                x=X,
                y=Y,
                z=depth_data,
                colorscale='Viridis',
                colorbar=dict(title='Depth (cm)'),
                contours=dict(
                    z=dict(
                        show=True,
                        usecolormap=True,
                        project_z=True
                    )
                )
            )
        ])
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X (pixels)',
                yaxis_title='Y (pixels)',
                zaxis_title='Depth (cm)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                aspectratio=dict(x=1, y=1, z=0.5)
            ),
            width=900,
            height=700
        )
        
        return fig
        
    def create_depth_heatmap(self, depth_map: np.ndarray,
                            mask: Optional[np.ndarray] = None,
                            title: str = "Depth Heatmap") -> go.Figure:
        """
        Create 2D heatmap of depth data.
        
        Args:
            depth_map: 2D depth map array
            mask: Optional binary mask
            title: Plot title
            
        Returns:
            plotly.graph_objects.Figure
        """
        if mask is not None:
            depth_data = depth_map.copy()
            depth_data[mask == 0] = np.nan
        else:
            depth_data = depth_map
            
        fig = go.Figure(data=go.Heatmap(
            z=depth_data,
            colorscale='Viridis',
            colorbar=dict(title='Depth (cm)')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='X (pixels)',
            yaxis_title='Y (pixels)',
            width=800,
            height=600
        )
        
        return fig
        
    def analyze_plate_depth(self, depth_map: np.ndarray,
                          plate_mask: np.ndarray) -> Dict:
        """
        Analyze depth values in plate region for validation.
        
        Args:
            depth_map: Depth map array
            plate_mask: Binary mask of plate region
            
        Returns:
            Dict containing analysis results
        """
        # Extract plate depth values
        plate_depths = depth_map[plate_mask > 0]
        
        if len(plate_depths) == 0:
            raise ValueError("No valid depth values in plate region")
            
        # Calculate statistics
        stats = {
            'mean_depth': float(np.mean(plate_depths)),
            'std_depth': float(np.std(plate_depths)),
            'min_depth': float(np.min(plate_depths)),
            'max_depth': float(np.max(plate_depths)),
            'num_points': int(len(plate_depths)),
            'expected_depth': self.expected_plate_distance,
            'depth_error': float(np.mean(plate_depths) - self.expected_plate_distance)
        }
        
        # Calculate planarity
        if len(plate_depths) > 3:
            planarity = self._calculate_planarity(depth_map, plate_mask)
            stats['planarity_error'] = float(planarity)
            
        return stats
        
    def _calculate_planarity(self, depth_map: np.ndarray, 
                           mask: np.ndarray) -> float:
        """Calculate RMSE from fitted plane."""
        # Get coordinates of valid points
        ys, xs = np.nonzero(mask)
        depths = depth_map[mask > 0]
        
        # Fit plane using least squares
        A = np.column_stack([xs, ys, np.ones_like(xs)])
        plane_params, _, _, _ = np.linalg.lstsq(A, depths, rcond=None)
        
        # Calculate error from plane
        fitted_depths = A @ plane_params
        rmse = np.sqrt(np.mean((depths - fitted_depths) ** 2))
        
        return rmse
        
    def create_depth_profile(self, depth_map: np.ndarray,
                           start_point: Tuple[int, int],
                           end_point: Tuple[int, int],
                           title: str = "Depth Profile") -> go.Figure:
        """
        Create line plot showing depth values along a line.
        
        Args:
            depth_map: Depth map array
            start_point: (x, y) starting point
            end_point: (x, y) ending point
            title: Plot title
            
        Returns:
            plotly.graph_objects.Figure
        """
        # Extract points along line
        num_points = 100
        x = np.linspace(start_point[0], end_point[0], num_points).astype(int)
        y = np.linspace(start_point[1], end_point[1], num_points).astype(int)
        
        # Get depth values
        depths = depth_map[y, x]
        
        # Create distance array
        distances = np.sqrt(
            (x - start_point[0])**2 + 
            (y - start_point[1])**2
        )
        
        fig = go.Figure(data=go.Scatter(
            x=distances,
            y=depths,
            mode='lines+markers',
            name='Depth Profile'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Distance along line (pixels)',
            yaxis_title='Depth (cm)',
            width=800,
            height=500
        )
        
        return fig
        
    def save_visualization(self, fig: go.Figure, 
                          output_path: Path,
                          filename: str) -> None:
        """Save visualization as HTML file."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig.write_html(str(output_dir / filename))
        logger.info(f"Saved visualization to {output_dir / filename}")

class ReconstructionVisualizer:
    """Visualization tools for 3D reconstruction analysis."""
    
    def __init__(self):
        """Initialize with default color scheme."""
        self.colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 
                      'rgb(44, 160, 44)', 'rgb(214, 39, 40)']
                      
    def plot_point_cloud(self, points: np.ndarray, 
                        title: str = "3D Point Cloud",
                        color: Optional[str] = None) -> go.Figure:
        """
        Create interactive 3D scatter plot of point cloud.
        
        Args:
            points: Nx3 array of 3D points
            title: Plot title
            color: Optional color for points
            
        Returns:
            plotly.graph_objects.Figure
        """
        fig = go.Figure(data=[go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=color or self.colors[0],
                opacity=0.8
            )
        )])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X (cm)',
                yaxis_title='Y (cm)',
                zaxis_title='Z (cm)',
                aspectmode='data'
            )
        )
        return fig
        
    def plot_multiple_objects(self, objects: Dict[str, np.ndarray],
                            title: str = "Multi-object Reconstruction") -> go.Figure:
        """
        Create interactive 3D plot of multiple point clouds.
        
        Args:
            objects: Dictionary mapping object names to point clouds
            title: Plot title
            
        Returns:
            plotly.graph_objects.Figure
        """
        fig = go.Figure()
        
        for i, (name, points) in enumerate(objects.items()):
            fig.add_trace(go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                name=name,
                marker=dict(
                    size=2,
                    color=self.colors[i % len(self.colors)],
                    opacity=0.8
                )
            ))
            
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X (cm)',
                yaxis_title='Y (cm)',
                zaxis_title='Z (cm)',
                aspectmode='data'
            )
        )
        return fig
        
    def plot_volume_estimation(self, points: np.ndarray, 
                             hull_vertices: np.ndarray,
                             volume: float,
                             title: Optional[str] = None) -> go.Figure:
        """
        Create interactive 3D visualization of volume estimation.
        
        Args:
            points: Nx3 array of 3D points
            hull_vertices: Indices of convex hull vertices
            volume: Calculated volume in cubic centimeters
            title: Optional plot title
            
        Returns:
            plotly.graph_objects.Figure
        """
        fig = go.Figure()
        
        # Original points
        fig.add_trace(go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            name='Points',
            marker=dict(size=2, color='blue', opacity=0.6)
        ))
        
        # Convex hull
        fig.add_trace(go.Mesh3d(
            x=points[hull_vertices, 0],
            y=points[hull_vertices, 1],
            z=points[hull_vertices, 2],
            opacity=0.3,
            color='red',
            name='Convex Hull'
        ))
        
        plot_title = title or f"Volume Estimation: {volume:.2f} cm³"
        fig.update_layout(
            title=plot_title,
            scene=dict(
                xaxis_title='X (cm)',
                yaxis_title='Y (cm)',
                zaxis_title='Z (cm)',
                aspectmode='data'
            )
        )
        return fig
        
    def save_visualization(self, fig: go.Figure,
                         output_dir: Path,
                         filename: str) -> None:
        """
        Save visualization as HTML file.
        
        Args:
            fig: Plotly figure object
            output_dir: Output directory path
            filename: Output filename (should end with .html)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / filename
        fig.write_html(str(output_path))
        logger.info(f"Saved visualization to {output_path}")
