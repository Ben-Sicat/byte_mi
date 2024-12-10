# M-PCAM Model

## Getting started
  - start a virtual environment
  
  to run the system ``` python run_volume_estimation.py``` 
  


## shit to improve on
- initialize the flask app
- add function to call api for the macronutrient database.




# TO RUN 

python run_volume_estimation --config test_config.json

endpoint is http://127.0.0.1:5000/get_volumes


{
  "results": [
    {
      "frame_id": "frame_1",
      "volumes": [
        {
          "object_name": "egg scrambled",
          "volume_cups": 0.5,
          "uncertainty_cups": 0.05
        },
        {
          "object_name": "oatmeal",
          "volume_cups": 1.0,
          "uncertainty_cups": 0.1
        }
      ],
      "nutrition": {
        "data": [
          {
            "food_name": "egg scrambled",
            "calories": 70,
            "protein": 6,
            "carbs": 1,
            "fat": 5
          },
          {
            "food_name": "oatmeal",
            "calories": 150,
            "protein": 5,
            "carbs": 27,
            "fat": 3
          }
        ]
      }
    }
  ]
}

```markdown
## Chapter IV: Results and Discussion

*Fig. 3. Controlled Test Case: Rice (0.67 cups) and Egg (0.5 cups)*

[Include actual rgb, mask, and 3D visualization images]

The system's performance was evaluated using a test case comprising rice and egg on a white plate. Fig. 3 presents the complete processing pipeline:

1) **Initial Capture and Processing:**
- RGB Input Image (Fig. 3a): Shows clear distinction between food items and plate
- Raw Depth Data (160x90): 
  - Stored in raw binary format
  - Depth values range: [318, 375]
  - Scaled using calibration factor: 0.0972
  - Final depth range in centimeters: [31.13, 33.55]

2) **Segmentation Performance:**
- Object Detection and Segmentation:
  - Plate area: 85,650 pixels (original), 4,005 pixels (aligned)
  - Rice area: 19,149 pixels (original), 899 pixels (aligned)
  - Egg area: 7,542 pixels (original), 347 pixels (aligned)
- Successful identification of distinct food items and plate boundary
- Clear separation between rice and egg regions

3) **Depth Processing Analysis:**
- Resolution handling:
  - Original depth capture: 160x90
  - Alignment with RGB: 480x640
  - Maintained aspect ratio during upscaling
- Depth Calibration:
  - Camera height: 33.0 cm
  - Plate surface height: 32.30 cm
  - Focal length: 105.54 pixels
  - Pixel size: 0.312687 cm/pixel

4) **Volume Estimation Results:**
For Rice:
- Raw depths range: [31.69, 33.55] cm
- Height range: [0.00, 1.86] cm
- Base area: 87.90 cm²
- Estimated volume: 0.66 cups (±0.07)
- Ground truth: 0.67 cups
- Accuracy: 98.5%

For Egg:
- Raw depths range: [31.13, 31.68] cm
- Height range: [0.00, 0.56] cm
- Base area: 33.93 cm²
- Estimated volume: 0.41 cups (±0.04)
- Ground truth: 0.5 cups
- Accuracy: 82%

5) **3D Visualization Analysis:**
The interactive 3D visualization confirms:
- Correct spatial relationships between food items
- Appropriate height differentials
- Clear object boundaries
- Reference plate surface as baseline

6) **System Limitations:**
- Raw depth data accessibility requires specific processing
- Edge cases in depth measurement near object boundaries
- Resolution constraints from initial 160x90 depth capture
- Noise in flat surface measurements
- Environmental dependencies:
  - Lighting conditions
  - Surface texture
  - Camera stability

The results demonstrate the system's capability to:
1. Process low-resolution depth data effectively
2. Maintain accuracy in volume estimation
3. Handle multiple food items simultaneously
4. Generate interactive 3D visualizations
5. Provide uncertainty estimates for measurements

The accuracy variation between rice (98.5%) and egg (82%) suggests better handling of spread-out items compared to compact foods, possibly due to more reliable height measurements across larger areas.
```
