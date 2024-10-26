# Food Volume Estimation from RGB Images

## Project Overview
A system for estimating food volume using only RGB images through depth estimation and 3D reconstruction.

### Core Objectives
- Convert RGB images to depth maps
- Use plate as reference object
- Apply pinhole camera model for 3D reconstruction
- Calculate food volumes from reconstructed 3D data

## Technical Pipeline

### 1. Image Processing Flow
```plaintext
RGB Image → RGBD Conversion → Depth Map → 3D Reconstruction → Volume Calculation
```

### 2. Key Components
- **RGB to Depth Conversion**: Using intensity and texture analysis
- **Segmentation**: Masks for individual food items
- **Calibration**: Plate as reference (25.5cm diameter, 1.5cm height)
- **3D Reconstruction**: Pinhole camera model implementation

## Current Implementation

### 1. Depth Estimation
```python
class PreprocessingPipeline:
    def __init__(self):
        self.camera_height = 33  # cm
        self.plate_height = 1.5  # cm
        self.plate_depth = 0.7   # cm

    def color_to_depth(self, rgbd_image, segmentation_mask):
        # Convert RGB to depth using:
        # - Intensity analysis
        # - Texture features
        # - Local depth relationships
        # Darker pixels = Higher elevation
```

### 2. 3D Reconstruction
```python
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
```

## Key Parameters

### Camera Configuration
```python
focal_length_mm = 7.0
sensor_width_mm = 4.8
pixel_size = 0.5  # cm per pixel
```

### Physical References
```python
plate_diameter = 25.5  # cm
plate_height = 1.5     # cm
plate_depth = 0.7      # cm
```

## Current Progress

### Achievements
1. **Depth Estimation**
   - RGB to depth conversion pipeline
   - Feature preservation in depth maps
   - Plate-relative height calibration

2. **3D Reconstruction**
   - Implemented pinhole camera model
   - World coordinate conversion
   - 3D point cloud visualization

3. **Volume Calculation**
   - Base area calculations
   - Height measurements
   - Initial volume estimation methods

### Challenges

1. **Depth Estimation Issues**
```python
# Current challenges:
# 1. Intensity to height relationship
# 2. Consistency across food types
# 3. Local vs global depth relationships
```

2. **Volume Calculation Problems**
```plaintext
Current Results:
Base Area (cm²): 5496.00
Average Height (cm): 1.70
Max Height (cm): 2.12
Volume (cm³): 0.00  # Issue with calculation
```

3. **Validation Challenges**
- Limited ground truth data
- Single reference object
- Camera parameter uncertainty

## Technical Decisions

### 1. Depth Conversion
```python
# Key principle:
depth_variation = 1.0 - (intensity - min_intensity) / (max_intensity - min_intensity)

# Height ranges:
base_height = plate_height + (plate_depth * 0.2)
max_height_variation = 2.5  # cm above plate
```

### 2. Volume Calculation Method
```python
def estimate_volume_from_mask(depth_map, mask, pixel_size, plate_height):
    # Current approach:
    valid_depths = depth_map[mask] - plate_height
    positive_depths = np.maximum(valid_depths, 0)
    pixel_area = pixel_size ** 2
    volume = np.sum(positive_depths) * pixel_area
```

## Proposed Improvements

### 1. Depth Estimation
```python
# Proposed enhancements:
- Better local feature preservation
- Improved texture analysis
- Adaptive height scaling
```

### 2. Volume Calculation
```python
# Proposed method:
def improved_volume_calculation():
    for pixel in object_mask:
        if depth > plate_height:
            column_volume = pixel_area * (depth - plate_height)
            total_volume += column_volume
```

### 3. Validation Methods
- Multiple reference objects
- Cross-validation with different views
- Known volume comparisons

## Next Steps

### Short Term
1. Fix volume calculation
2. Improve depth estimation accuracy
3. Implement validation methods

### Long Term
1. Multiple reference object support
2. Advanced texture analysis
3. Uncertainty estimation

## Open Questions

### Technical Considerations
1. How to improve intensity-depth relationship?
2. What additional features for depth estimation?
3. How to validate without ground truth?

### Implementation Decisions
1. Choice of volume calculation method
2. Depth map refinement approach
3. Calibration strategy

## Additional Notes

### Dependencies
```python
import numpy as np
import cv2
import plotly.graph_objects as go
from scipy.ndimage import uniform_filter, gaussian_filter
```

### File Structure
```plaintext
src/
├── preprocessing/
│   └── preprocess.py
└── reconstruction/
    └── 3d_reconstruction.py
```

### Usage Example
```python
# Basic usage
preprocessor = PreprocessingPipeline(data_dir, output_dir)
depth_map = preprocessor.process_image(rgb_filename, image_id)
volume = estimate_volume_from_mask(depth_map, mask, intrinsic_params, pixel_size)
```
