import sys
from pathlib import Path
import numpy as np
import pytest
import trimesh
import json
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.reconstruction.volume_calculator import VolumeCalculator
from src.reconstruction.mesh_generator import MeshGenerator
from src.reconstruction.point_cloud import PointCloud
from src.preprocessing.preprocessing import PreprocessingPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def preprocessed_data():
    """Load preprocessed data from pipeline"""
    try:
        # Load test configuration
        with open('test_config.json', 'r') as f:
            config = json.load(f)
            
        # Get the first frame_id
        frame_id = config['frame_ids'][0]
        
        # Initialize and run pipeline
        pipeline = PreprocessingPipeline(config)
        result = pipeline.process_single_image(frame_id)
        
        return result
        
    except Exception as e:
        logger.error(f"Error loading preprocessing output: {str(e)}")
        raise

class TestVolumeCalculator:
    def test_food_volume_estimation(self, preprocessed_data):
        """Test volume calculation for food items"""
        pc = PointCloud(preprocessed_data['intrinsic_params'])
        calc = VolumeCalculator()
        
        results = {}
        
        for obj_name, obj_data in preprocessed_data['processed_objects'].items():
            # Skip plate
            if obj_name == 'plate':
                continue
                
            # Generate point cloud
            points = pc.depth_to_point_cloud(
                preprocessed_data['depth'],
                obj_data['mask']
            )
            
            # Calculate volume using different methods
            volumes = {}
            for method in ['convex_hull', 'grid', 'alpha_shape']:
                try:
                    result = calc.calculate_volume(points, method=method)
                    volumes[method] = result
                except Exception as e:
                    logger.warning(f"Method {method} failed for {obj_name}: {str(e)}")
                    
            results[obj_name] = volumes
            
            logger.info(f"\nVolume estimation for {obj_name}:")
            for method, result in volumes.items():
                logger.info(
                    f"{method}: {result['volume_cm3']:.2f}cm³ "
                    f"(±{result['uncertainty_cm3']:.2f}) "
                    f"= {result['volume_cups']:.2f} cups "
                    f"(±{result['uncertainty_cups']:.2f})"
                )
        
        assert len(results) > 0

    def test_method_consistency(self, preprocessed_data):
        """Test consistency between different volume calculation methods"""
        pc = PointCloud(preprocessed_data['intrinsic_params'])
        calc = VolumeCalculator()
        
        for obj_name, obj_data in preprocessed_data['processed_objects'].items():
            if obj_name == 'plate':
                continue
                
            points = pc.depth_to_point_cloud(
                preprocessed_data['depth'],
                obj_data['mask']
            )
            
            # Calculate volumes
            convex_result = calc.calculate_volume(points, 'convex_hull')
            grid_result = calc.calculate_volume(points, 'grid')
            
            # Check consistency within reasonable margin (50%)
            assert np.isclose(
                convex_result['volume_cm3'],
                grid_result['volume_cm3'],
                rtol=0.5
            )

class TestMeshGenerator:
    def test_food_mesh_generation(self, preprocessed_data):
        """Test mesh generation for food items"""
        pc = PointCloud(preprocessed_data['intrinsic_params'])
        generator = MeshGenerator()
        
        results = {}
        
        for obj_name, obj_data in preprocessed_data['processed_objects'].items():
            if obj_name == 'plate':
                continue
                
            # Generate point cloud
            points = pc.depth_to_point_cloud(
                preprocessed_data['depth'],
                obj_data['mask']
            )
            
            # Generate meshes using different methods
            meshes = {}
            for method in ['convex', 'surface', 'alpha']:
                try:
                    mesh = generator.generate_mesh(points, method=method)
                    validation = generator.validate_mesh(mesh)
                    meshes[method] = {
                        'mesh': mesh,
                        'validation': validation
                    }
                except Exception as e:
                    logger.warning(f"Method {method} failed for {obj_name}: {str(e)}")
                    
            results[obj_name] = meshes
            
            logger.info(f"\nMesh generation for {obj_name}:")
            for method, result in meshes.items():
                mesh = result['mesh']
                validation = result['validation']
                logger.info(
                    f"{method}: {len(mesh.vertices)} vertices, "
                    f"{len(mesh.faces)} faces\n"
                    f"Validation: {validation}"
                )
        
        assert len(results) > 0

    def test_mesh_volume_consistency(self, preprocessed_data):
        """Test consistency between mesh volume and direct calculation"""
        pc = PointCloud(preprocessed_data['intrinsic_params'])
        generator = MeshGenerator()
        calc = VolumeCalculator()
        
        for obj_name, obj_data in preprocessed_data['processed_objects'].items():
            if obj_name == 'plate':
                continue
                
            points = pc.depth_to_point_cloud(
                preprocessed_data['depth'],
                obj_data['mask']
            )
            
            # Calculate volume directly
            volume_result = calc.calculate_volume(points, 'convex_hull')
            
            # Generate mesh and calculate its volume
            mesh = generator.generate_mesh(points, 'convex')
            mesh_volume = mesh.volume
            
            # Compare volumes (within 10% for convex hull method)
            assert np.isclose(
                volume_result['volume_cm3'],
                mesh_volume,
                rtol=0.1
            )

def test_full_reconstruction_pipeline(preprocessed_data):
    """Test the complete reconstruction pipeline"""
    pc = PointCloud(preprocessed_data['intrinsic_params'])
    generator = MeshGenerator()
    calc = VolumeCalculator()
    
    reconstruction_results = {}
    
    for obj_name, obj_data in preprocessed_data['processed_objects'].items():
        if obj_name == 'plate':
            continue
            
        try:
            # 1. Generate point cloud
            points = pc.depth_to_point_cloud(
                preprocessed_data['depth'],
                obj_data['mask']
            )
            
            # 2. Generate mesh
            mesh = generator.generate_mesh(points, 'surface')
            validation = generator.validate_mesh(mesh)
            
            # 3. Calculate volume
            volume_result = calc.calculate_volume(points, 'convex_hull')
            
            reconstruction_results[obj_name] = {
                'num_points': len(points),
                'mesh_vertices': len(mesh.vertices),
                'mesh_faces': len(mesh.faces),
                'mesh_validation': validation,
                'volume_cm3': volume_result['volume_cm3'],
                'volume_cups': volume_result['volume_cups'],
                'uncertainty_cups': volume_result['uncertainty_cups']
            }
            
        except Exception as e:
            logger.error(f"Reconstruction failed for {obj_name}: {str(e)}")
            
    logger.info("\nFull reconstruction results:")
    for obj_name, result in reconstruction_results.items():
        logger.info(f"\n{obj_name}:")
        logger.info(f"Points: {result['num_points']}")
        logger.info(f"Mesh: {result['mesh_vertices']} vertices, {result['mesh_faces']} faces")
        logger.info(f"Volume: {result['volume_cups']:.2f} cups (±{result['uncertainty_cups']:.2f})")
        logger.info(f"Mesh validation: {result['mesh_validation']}")
        
    assert len(reconstruction_results) > 0
