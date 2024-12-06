import numpy as np
from typing import Dict, Optional, Tuple
import logging
import trimesh
from scipy.spatial import Delaunay
from sklearn.decomposition import PCA as skPCA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MeshGenerator:
    def __init__(self, smoothing_factor: float = 0.5):
        self.smoothing_factor = smoothing_factor
        
    def _surface_reconstruction(self, points: np.ndarray) -> trimesh.Trimesh:
        """Generate mesh using surface reconstruction."""
        pca = skPCA(n_components=2)
        points_2d = pca.fit_transform(points)
        
        tri = Delaunay(points_2d)
        
        mesh = trimesh.Trimesh(
            vertices=points,
            faces=tri.simplices
        )
        
        return mesh
    def generate_mesh(self, points: np.ndarray, 
                     method: str = 'surface') -> trimesh.Trimesh:
        """
        Generate mesh from point cloud.
        """
        try:
            if method == 'convex':
                mesh = self._convex_hull_mesh(points)
            elif method == 'surface':
                mesh = self._surface_reconstruction(points)
            elif method == 'alpha':
                mesh = self._alpha_shape_mesh(points)
            else:
                raise ValueError(f"Unknown method: {method}")
                
            # Apply smoothing
            if self.smoothing_factor > 0:
                mesh = self._smooth_mesh(mesh)
                
            logger.info(
                f"Generated mesh using {method} method: "
                f"{len(mesh.vertices)} vertices, {len(mesh.faces)} faces"
            )
            
            return mesh
            
        except Exception as e:
            logger.error(f"Mesh generation failed: {str(e)}")
            raise
            
    def _convex_hull_mesh(self, points: np.ndarray) -> trimesh.Trimesh:
        """Generate mesh using convex hull."""
        mesh = trimesh.Trimesh(vertices=points)
        return mesh.convex_hull
        
    def _surface_reconstruction(self, points: np.ndarray) -> trimesh.Trimesh:
        """Generate mesh using surface reconstruction."""
        # Project points to 2D for triangulation
        pca = trimesh.transformations.PCA(points)
        points_2d = points @ pca[:2].T
        
        # Create triangulation
        tri = Delaunay(points_2d)
        
        # Create mesh
        mesh = trimesh.Trimesh(
            vertices=points,
            faces=tri.simplices
        )
        
        return mesh
        
    def _alpha_shape_mesh(self, points: np.ndarray, 
                         alpha: float = None) -> trimesh.Trimesh:
        """Generate mesh using alpha shape."""
        if alpha is None:
            # Estimate alpha based on point density
            bbox_volume = np.prod(np.ptp(points, axis=0))
            point_density = len(points) / bbox_volume
            alpha = 1 / np.sqrt(point_density)
            
        mesh = trimesh.Trimesh(vertices=points)
        hull = mesh.convex_hull
        alpha_mesh = mesh.subdivide().intersection(hull)
        
        return alpha_mesh
        
    def _smooth_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Apply Laplacian smoothing to mesh."""
        # Create copy to avoid modifying original
        smoothed = mesh.copy()
        
        # Apply Laplacian smoothing
        factor = self.smoothing_factor
        vertices = smoothed.vertices
        adjacency = trimesh.graph.vertex_adjacency_graph(smoothed)
        
        for _ in range(3):  # Number of smoothing iterations
            new_vertices = vertices.copy()
            for i in range(len(vertices)):
                neighbors = adjacency[i].keys()
                if neighbors:
                    centroid = np.mean([vertices[j] for j in neighbors], axis=0)
                    new_vertices[i] += factor * (centroid - vertices[i])
            vertices = new_vertices
            
        smoothed.vertices = vertices
        return smoothed
        
    def validate_mesh(self, mesh: trimesh.Trimesh) -> Dict[str, bool]:
        """
        Validate mesh quality and properties.
        
        Args:
            mesh: trimesh.Trimesh object
            
        Returns:
            Dictionary of validation results
        """
        results = {
            'is_watertight': mesh.is_watertight,
            'is_winding_consistent': mesh.is_winding_consistent,
            'has_degenerate_faces': len(mesh.degenerate_faces) > 0,
            'has_duplicate_faces': len(mesh.duplicate_faces) > 0,
            'has_infinite_values': not np.all(np.isfinite(mesh.vertices))
        }
        
        logger.info(f"Mesh validation results: {results}")
        return results
