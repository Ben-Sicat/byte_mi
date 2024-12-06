import sys
from pathlib import Path
import numpy as np
import pytest
import json
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.reconstruction.point_cloud import PointCloud
from src.preprocessing.preprocessing import PreprocessingPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def preprocessing_output():
    """Load preprocessed data from the pipeline"""
    try:
        # Load test configuration
        with open('test_config.json', 'r') as f:
            config = json.load(f)
            
        # Get the first frame_id
        frame_id = config['frame_ids'][0]
        
        # Initialize pipeline
        pipeline = PreprocessingPipeline(config)
        
        # Process single image
        result = pipeline.process_single_image(frame_id)
        return result
        
    except Exception as e:
        logger.error(f"Error loading preprocessing output: {str(e)}")
        raise

def test_point_cloud_with_preprocessed_data(preprocessing_output):
    """Test point cloud generation using preprocessed data"""
    # Initialize point cloud with calibrated parameters
    pc = PointCloud(preprocessing_output['intrinsic_params'])
    
    # Generate point cloud from depth data
    points = pc.depth_to_point_cloud(
        preprocessing_output['depth'],
        None  # Full depth map
    )
    
    assert isinstance(points, np.ndarray)
    assert points.shape[1] == 3  # X, Y, Z coordinates
    assert not np.any(np.isnan(points))
    assert not np.any(np.isinf(points))
    
    logger.info(f"Generated point cloud with {len(points)} points")

def test_object_specific_point_clouds(preprocessing_output):
    """Test generating point clouds for individual objects"""
    pc = PointCloud(preprocessing_output['intrinsic_params'])
    
    for obj_name, obj_data in preprocessing_output['processed_objects'].items():
        # Generate point cloud for specific object
        obj_points = pc.depth_to_point_cloud(
            preprocessing_output['depth'],
            obj_data['mask']
        )
        
        assert len(obj_points) > 0
        assert not np.any(np.isnan(obj_points))
        
        # Calculate basic measurements
        dimensions = pc.get_dimensions(obj_points)
        assert all(v > 0 for v in dimensions.values())
        
        # Basic sanity checks for plate
        if obj_name == 'plate':
            # Check if dimensions roughly match expected plate size
            plate_diameter = preprocessing_output['intrinsic_params']['reference_object']['diameter']
            max_dim = max(dimensions['length'], dimensions['width'])
            assert np.isclose(max_dim, plate_diameter, rtol=0.2)
            
        logger.info(f"Processed object {obj_name}: {len(obj_points)} points")

def test_volume_estimation_with_real_data(preprocessing_output):
    """Test volume estimation with preprocessed data"""
    pc = PointCloud(preprocessing_output['intrinsic_params'])
    
    for obj_name, obj_data in preprocessing_output['processed_objects'].items():
        # Skip plate (reference object)
        if obj_name == 'plate':
            continue
            
        # Generate point cloud
        points = pc.depth_to_point_cloud(
            preprocessing_output['depth'],
            obj_data['mask']
        )
        
        # Calculate volume
        volume = pc.estimate_volume(points)
        assert volume > 0
        
        # Calculate surface area
        surface_area = pc.calculate_surface_area(points)
        assert surface_area > 0
        
        logger.info(
            f"Object {obj_name} measurements:\n"
            f"Volume: {volume:.2f} cm³\n"
            f"Surface Area: {surface_area:.2f} cm²"
        )

def test_data_consistency(preprocessing_output):
    """Test consistency between depth and mask data"""
    depth_shape = preprocessing_output['depth'].shape
    
    for obj_data in preprocessing_output['processed_objects'].values():
        assert obj_data['mask'].shape == depth_shape
        
        # Check if mask and depth alignment makes sense
        masked_depth = preprocessing_output['depth'][obj_data['mask'] > 0]
        assert len(masked_depth) > 0
        assert np.all(masked_depth > 0)  # No invalid depth values in mask
def test_reference_object_validation(preprocessing_output):
    """Validate measurements using reference plate"""
    pc = PointCloud(preprocessing_output['intrinsic_params'])
    
    # Get plate data
    plate_data = next(
        (data for name, data in preprocessing_output['processed_objects'].items() 
         if name == 'plate'),
        None
    )
    
    if plate_data is not None:
        # Generate plate point cloud
        plate_points = pc.depth_to_point_cloud(
            preprocessing_output['depth'],
            plate_data['mask']
        )
        
        # Get plate dimensions
        dims = pc.get_dimensions(plate_points)
        
        # Get plate center height
        center_height = preprocessing_output['intrinsic_params']['camera_height']
        center_depth = preprocessing_output['depth'][
            preprocessing_output['intrinsic_params']['reference_object']['center_pixels'][1],
            preprocessing_output['intrinsic_params']['reference_object']['center_pixels'][0]
        ]
        
        # Check reference height
        assert np.isclose(center_depth, center_height, rtol=0.2)
        
        # Check against known plate diameter with larger tolerance due to perspective effects
        plate_diameter = preprocessing_output['intrinsic_params']['reference_object']['diameter']
        max_dim = max(dims['length'], dims['width'])
        assert np.isclose(max_dim, plate_diameter, rtol=0.3)
        
        logger.info(
            f"Plate validation:\n"
            f"Expected diameter: {plate_diameter:.2f}cm\n"
            f"Measured diameter: {max_dim:.2f}cm\n"
            f"Center height: {center_depth:.2f}cm"
        )
